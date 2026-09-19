"""Train the label quality assurance model on Pascal VOC 2012.

The small set is the Pascal VOC 2012 validation split (5823 images): 90% of
it (5241 images) trains the model and 10% (582 images) is held out for
model selection and the refit (refit.py). Each epoch passes through every
training image twice — once requesting a bad candidate and once a good
one, drawn anew every epoch.

The network is neural_network.CorrNet (README item 2). The defaults are
the release recipe, except the per-cell head and its two losses, which
run.sh adds (--raster_head --box_weight 0.5 --cls_weight 0.5, README
item 4): AdamW (lr 1e-3, betas (0.9, 0.999), eps 1e-8, weight decay 1e-2,
as in Table 8.4) with the pretrained image branch fine-tuned at a tenth of
the rate (README item 5), two warmup epochs then cosine decay over 160
epochs, an effective batch of 64 (32 x 2 accumulated) with BatchNorm
statistics per group of 8 (the pretrained image branch keeps its own),
mixed precision, and a per-step weight EMA (0.998). Every epoch the EMA
weights are scored on a fixed validation set drawn as evaluate.py draws
the large set (data_loader.fixed_rows, rounds 0-3: 4656 candidates); all
epochs run (--patience 0) and the epoch with the highest ROC-AUC is kept
as best_model.pth (README item 9).

Training candidates: the Listing 8.11 augmentations on one sample in ten;
half of the good candidates are the ground truth verbatim (README item 1);
of the negatives, 10% pair a good label with another training image
(README item 8), 10% of the rest come from a detector, and of the
generated ones half are image-hard (data_loader.corrupt_mixed, README
item 7), half the thesis generator's.

    python3 train.py --raster_head --box_weight 0.5 --cls_weight 0.5 \\
        --workers 20 --seed 0 --out artifacts --resume
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from data_loader import Data, P_EXACT_GT, build_context, fixed_rows
from neural_network import CorrNet, ghostify


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def worker_init(wid):
    s = (torch.initial_seed() + wid) % (2 ** 31)
    random.seed(s)
    np.random.seed(s)
    # one cv2 thread per worker: the default per-process pool times a
    # large worker count oversubscribes the host; output is unaffected
    import cv2
    cv2.setNumThreads(0)


def pool_split(pool, seed, val_split, limit=0):
    """The small set's files (images with at least one box), shuffled with
    seed, split into (validation, training)."""
    pool_dir = os.path.expanduser(pool)
    files = sorted(os.path.join(pool_dir, f) for f in os.listdir(pool_dir)
                   if f.endswith(".npy"))
    files = [f for f in files
             if any(len(v) for v in
                    np.load(f, allow_pickle=True).item().values())]
    random.Random(seed).shuffle(files)
    if limit:
        files = files[:limit]
    n_val = max(1, int(val_split * len(files)))
    return files[:n_val], files[n_val:]


def evaluate_split(model, loader, device, criterion):
    model.eval()
    losses, probs, labels = [], [], []
    with torch.no_grad():
        for img, planes, y in loader:
            img = img.float().to(device)
            planes = planes.float().to(device)
            y = y.long().to(device)
            logits = model(img, planes).float()
            losses.append(criterion(logits, y).item() * y.size(0))
            probs.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
            labels.extend(y.cpu().tolist())
    loss = sum(losses) / max(1, len(labels))
    preds = [int(p >= 0.5) for p in probs]
    acc = sum(int(p == l) for p, l in zip(preds, labels)) / max(1, len(labels))
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return loss, acc, auc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool",
                    default="~/fiftyone/voc-2012/validation/processed_annotations",
                    help="small-set annotation pool (the VOC validation split)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2,
                    help="batches accumulated per optimizer step")
    ap.add_argument("--ghost_bn", type=int, default=8,
                    help="BatchNorm statistics per group of this many "
                         "samples (0 = the whole batch)")
    ap.add_argument("--no_amp", action="store_true",
                    help="train in fp32 instead of mixed precision")
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--img_lr_mult", type=float, default=0.1,
                    help="learning-rate factor of the pretrained image branch")
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--ema_decay", type=float, default=0.998,
                    help="per-step weight EMA; validation and the saved "
                         "models use the EMA weights (0 = off)")
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--val_rounds", type=int, default=4,
                    help="draw rounds of the fixed validation set")
    ap.add_argument("--detector_share", type=float, default=0.1)
    ap.add_argument("--p_exact_gt", type=float, default=P_EXACT_GT,
                    help="probability that a good training candidate is "
                         "the ground truth verbatim (README)")
    ap.add_argument("--ih_share", type=float, default=0.5,
                    help="share of generated negatives that are image-hard")
    ap.add_argument("--swap_share", type=float, default=0.1,
                    help="share of negatives pairing a good label with "
                         "another training image")
    ap.add_argument("--raster_head", action="store_true",
                    help="add the per-cell head that reads the label "
                         "planes (neural_network.py)")
    ap.add_argument("--box_weight", type=float, default=0.0,
                    help="weight of --raster_head's per-cell support loss "
                         "against the per-box form of Equation 7.1")
    ap.add_argument("--cls_weight", type=float, default=0.0,
                    help="weight of --raster_head's per-cell class loss "
                         "against the ground-truth class map")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after this many epochs without a validation "
                         "ROC-AUC improvement (0 = off: train all --epochs)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=2,
                    help="batches prefetched per worker")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/checkpoint.pth (written every "
                         "epoch); off by default, and the checkpoint is "
                         "ignored unless this flag is passed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_files", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    if args.ghost_bn and args.batch_size % args.ghost_bn:
        ap.error(f"--batch_size must be a multiple of --ghost_bn "
                 f"{args.ghost_bn}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    val_files, train_files = pool_split(args.pool, args.seed, args.val_split,
                                        args.limit_files)
    print(f"{len(train_files)} train / {len(val_files)} validation images",
          flush=True)
    context = build_context(train_files) if args.ih_share > 0 else None
    train_rows = [(f, r) for f in train_files for r in (0, 1)]
    train_ds = Data(train_rows, augment=True,
                    detector_share=args.detector_share,
                    p_exact_gt=args.p_exact_gt,
                    cell_targets=args.raster_head, ih_share=args.ih_share,
                    context=context, swap_share=args.swap_share)
    # validation: the evaluation draw (evaluate.py) on fixed rows, the same
    # candidates every epoch (README item 9)
    val_ds = Data(fixed_rows(val_files, args.val_rounds, args.seed),
                  augment=False, detector_share=args.detector_share)
    print(f"validation: {len(val_ds)} candidates ({args.val_rounds} rounds, "
          f"evaluation draw)", flush=True)
    loader_kw = dict(num_workers=args.workers, worker_init_fn=worker_init,
                     persistent_workers=args.workers > 0,
                     pin_memory=device.type == "cuda")
    if args.workers > 0:
        loader_kw["prefetch_factor"] = args.prefetch
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        **loader_kw)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=64, shuffle=False, **loader_kw)

    model = CorrNet(raster_head=args.raster_head)
    if args.ghost_bn:
        ghostify(model, args.ghost_bn)
    model.to(device)
    # the pretrained image branch trains at --img_lr_mult x the rate
    # (README item 5)
    image_ids = {id(p) for p in model.model_1.parameters()}
    groups = [{"params": [p for p in model.parameters()
                          if id(p) not in image_ids]},
              {"params": list(model.model_1.parameters()),
               "lr": args.lr * args.img_lr_mult}]
    optimizer = torch.optim.AdamW(
        groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=args.weight_decay)
    steps_per_epoch = -(-len(train_loader) // args.accum)
    warmup = args.warmup_epochs * steps_per_epoch
    total = args.epochs * steps_per_epoch

    def lr_at(step):
        # linear warmup, then cosine decay to 0 at the last step
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()
    print(f"AdamW lr {args.lr:g} (image branch x{args.img_lr_mult:g}), "
          f"weight decay {args.weight_decay:g}; batch {args.batch_size} x "
          f"{args.accum}, ghost BN {args.ghost_bn}, amp={use_amp}; "
          f"{steps_per_epoch} steps/epoch, warmup {args.warmup_epochs}, "
          f"cosine over {args.epochs} epochs; EMA {args.ema_decay:g}",
          flush=True)

    # per-step weight EMA; validation and the saved models use it (README
    # item 9)
    ema = None
    if args.ema_decay > 0:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def ema_update():
        if ema is None:
            return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(args.ema_decay).add_(
                        v, alpha=1 - args.ema_decay)
                else:
                    ema[k].copy_(v)

    def step():
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        sched.step()
        ema_update()

    os.makedirs(args.out, exist_ok=True)
    best_path = os.path.join(args.out, "best_model.pth")
    ckpt_path = os.path.join(args.out, "checkpoint.pth")
    history, best_auc, best_epoch, start_epoch = [], -1.0, 0, 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["sched"])
        scaler.load_state_dict(ckpt["scaler"])
        if ema is not None:
            ema = ckpt["ema"]
        history, best_auc = ckpt["history"], ckpt["best_auc"]
        best_epoch, start_epoch = ckpt["best_epoch"], ckpt["epoch"]
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        # map_location=device put the saved RNG states on the GPU; both
        # setters take CPU byte tensors only
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        if torch.cuda.is_available() and ckpt["rng"].get("cuda") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng"]["cuda"]])
        print(f"resumed from {ckpt_path} at epoch {start_epoch + 1}",
              flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, steps, t0 = 0.0, 0, time.time()
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            img, planes, y = batch[:3]
            img = img.float().to(device, non_blocking=True)
            planes = planes.float().to(device, non_blocking=True)
            y = y.long().to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if args.raster_head and (args.box_weight > 0
                                         or args.cls_weight > 0):
                    logits, sup, cls_map = model(img, planes, aux=True)
                    loss = criterion(logits.float(), y)
                    T = batch[3].to(device, non_blocking=True)
                    if args.box_weight > 0:
                        # the per-cell losses (README item 4): support of
                        # each claimed (class plane, cell) against that
                        # cell's box_good (255 = nothing drawn)
                        claimed = T != 255
                        if claimed.any():
                            loss = loss + args.box_weight * \
                                F.binary_cross_entropy_with_logits(
                                    sup[claimed], T[claimed].float())
                    if args.cls_weight > 0:
                        loss = loss + args.cls_weight * F.cross_entropy(
                            cls_map.float(),
                            batch[4].long().to(device, non_blocking=True))
                else:
                    loss = criterion(model(img, planes).float(), y)
            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                step()
            running += loss.item()
            steps += 1
        if steps % args.accum:
            step()
        raw = None
        if ema is not None:
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema)
        val_loss, val_acc, val_auc = evaluate_split(model, val_loader, device,
                                                    criterion)
        star = ""
        is_best = val_auc > best_auc
        if is_best:
            best_auc, best_epoch = val_auc, epoch + 1
            star = "  *best*"
        history.append({"epoch": epoch + 1, "loss": running / max(1, steps),
                        "val_loss": val_loss, "val_acc": val_acc,
                        "val_auc": val_auc})
        # Every file is written to a temporary name and renamed into place,
        # so an interruption never leaves a truncated file; the resume state
        # (which carries best_auc) lands before best_model.pth so a resumed
        # run cannot overwrite a better model with a worse one.
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if raw is not None:
            model.load_state_dict(raw)
        atomic_save({"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "ema": ema,
                     "history": history, "best_auc": best_auc,
                     "best_epoch": best_epoch, "epoch": epoch + 1,
                     "rng": {"python": random.getstate(),
                             "numpy": np.random.get_state(),
                             "torch": torch.get_rng_state(),
                             "cuda": (torch.cuda.get_rng_state_all()
                                      if torch.cuda.is_available() else None)}},
                    ckpt_path)
        atomic_save(saved, os.path.join(args.out, "model.pth"))
        if is_best:
            atomic_save(saved, best_path)
        print(f"epoch {epoch + 1}/{args.epochs} "
              f"loss={running / max(1, steps):.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.3f} val_auc={val_auc:.4f}{star} "
              f"({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(args.out, "training_log.json"), "w") as f:
            json.dump({"args": vars(args), "best_val_auc": best_auc,
                       "best_epoch": best_epoch, "history": history},
                      f, indent=2)
            f.write("\n")
        if args.patience and epoch + 1 - best_epoch >= args.patience:
            print(f"early stop: no validation ROC-AUC improvement in "
                  f"{args.patience} epochs", flush=True)
            break

    print(f"best validation ROC-AUC {best_auc:.4f} (epoch {best_epoch}); "
          f"artifacts in {args.out}/")


if __name__ == "__main__":
    main()
