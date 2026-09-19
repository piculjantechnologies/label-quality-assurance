"""Trainer for the unified implementation (QA_DATASET=coco|voc).

Dataset-agnostic: the dataset side (pool, classes, input resolution,
fusion grid, validation share) comes from `dataset.py`, the candidates and
rendering from `qa_data.py`, the model from `qa_model.py`.

Recipe (carried by `run_qa.sh`; the argparse defaults below are not it):
AdamW 1e-3 / weight decay 1e-2, cosine schedule with 2 warmup epochs,
effective batch 64 run as one physical batch with GhostBatchNorm
statistics per sub-batch of 8 (`FAITHFUL=1` runs the 8 x 8 accumulated
form), AMP, 60 epochs, seed 0, ResNet-50 image trunk, per-box head.
Augmentation on (flip / mild affine / photometric, p 0.5). Negatives from
the unified mixer (image-swap 10%, drawn first; of the rest, detector
10%, image-hard 45%, severity ops 45% -- about 10 / 9 / 40 / 40%
overall; half of the image-hard and severity negatives stack 1-3 extra
errors).
Auxiliary box-class supervision at weight 0.3. A weight-EMA (0.998)
candidate is tracked alongside the raw weights; the EMA candidate is
calibrated with qa_calibrate.py.

The generator's configuration (the positive sampler's p_exact / p_edge,
the training-pool box bank and class co-occurrence) is carried on the
dataset objects and installed in every DataLoader worker by worker_init,
so the loaders draw the same samples under every multiprocessing start
method (fork, spawn, forkserver).

    QA_DATASET=voc VOC_POOL=... ./run_qa.sh              # writes artifacts_voc/

The same run by hand (the command line run_qa.sh executes by default,
WORKERS defaulting to 20; with FAITHFUL=1 the script passes
`--batch_size 8 --accum 8` in place of
`--batch_size 64 --accum 1 --ghost_bn 8 --workers 20 --cache_images`,
leaving --workers at the argparse default of 4):

    QA_DATASET=voc VOC_POOL=... python3 qa_train.py \
        --epochs 60 --batch_size 64 --accum 1 --ghost_bn 8 --workers 20 \
        --cache_images --amp \
        --lr 1e-3 --weight_decay 1e-2 --warmup_epochs 2 \
        --finetune_mult 0.1 --freeze_through layer2 \
        --ema_decay 0.998 --aux_weight 0.3 \
        --p_exact 0.5 --detector_share 0.1 --ih_share 0.5 --compose 0.5 \
        --severity_mix balanced --aug_prob 0.5 --seed 0 \
        --p_edge 0.25 --swap_share 0.1 \
        --per_box_weight 0.3 --margin_weight 0.3 \
        --image_trunk resnet50 \
        --out artifacts_voc --resume
"""

import argparse
import functools
import json
import os
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dataset as ds_conf
import qa_data
from qa_model import FusionNet


def recipe_state():
    """Snapshot of qa_data's process-level configuration, taken once
    main() has applied it: the positive sampler's keyword overrides (the
    p_exact / p_edge partial over good_label, stored as its keywords) and
    the image-hard context built by build_context (the training-pool box
    bank and class co-occurrence).

    The datasets carry this dict so that DataLoader workers started from
    a fresh interpreter -- the spawn and forkserver start methods: macOS,
    and Linux from Python 3.14 -- are brought to the parent's state by
    worker_init instead of running qa_data at its module defaults (p_edge
    0, no bank, uniform absent-class choice), which would draw a
    different candidate population without any error. Everything here
    pickles: plain tables, and the partial is stored as its keywords
    because a partial over a rebound module attribute does not.
    Fork-started workers inherit the state and re-apply the same values.
    qa_ablation.py and qa_calibrate.py rebuild their validation
    populations with the same helpers.
    """
    good = qa_data.good_label
    return {
        "good_kw": (dict(good.keywords)
                    if isinstance(good, functools.partial) else None),
        "context": (qa_data._BANK, qa_data._COOC),
    }


def install_recipe(state):
    """Apply a recipe_state() dict to this process's qa_data module."""
    if state is None:
        return
    good = qa_data.good_label
    if isinstance(good, functools.partial):
        good = good.func
    qa_data.good_label = (functools.partial(good, **state["good_kw"])
                          if state["good_kw"] else good)
    qa_data._BANK, qa_data._COOC = state["context"]


def worker_init(worker_id):
    """DataLoader worker_init_fn: one cv2 thread per worker, then the
    generator recipe carried by the worker's dataset (recipe_state)."""
    qa_data.cv2_single_thread_worker(worker_id)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        install_recipe(getattr(info.dataset, "recipe", None))


def evaluate_loader(model, loader, device, amp):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            img, planes, idx, y = batch[:4]
            img, planes = qa_data.to_dense_batch(img, planes, idx, device)
            roi = batch[-1][:, :6].to(device) if model.per_box else None
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(planes, img, boxes=roi)
            scores += torch.softmax(logits.float(), 1)[:, 1].tolist()
            labels += y.tolist()
    model.train()
    auc = roc_auc_score(labels, scores)
    acc = float(np.mean((np.array(scores) >= 0.5) == np.array(labels)))
    return auc, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--finetune_mult", type=float, default=0.1)
    ap.add_argument("--freeze_through", default="layer2")
    ap.add_argument("--ema_decay", type=float, default=0.998)
    ap.add_argument("--aux_weight", type=float, default=0.3)
    ap.add_argument("--p_exact", type=float, default=0.5)
    ap.add_argument("--p_edge", type=float, default=0.0,
                    help="share of jittered positives drawn near the band "
                         "boundary (0 = plain uniform draw; run_qa.sh "
                         "uses 0.25)")
    ap.add_argument("--detector_share", type=float, default=0.1)
    ap.add_argument("--ih_share", type=float, default=0.5)
    ap.add_argument("--compose", type=float, default=0.5,
                    help="probability a negative stacks 1-3 extra errors")
    ap.add_argument("--swap_share", type=float, default=0.0,
                    help="share of negatives pairing a good label with a "
                         "different pool image (grounding negatives)")
    ap.add_argument("--per_box_weight", type=float, default=0.0,
                    help="> 0 enables the per-box verification head and "
                         "weights its support BCE")
    ap.add_argument("--margin_weight", type=float, default=0.0,
                    help="> 0 enables the head's signed band-margin "
                         "regression and weights its smooth-L1")
    ap.add_argument("--ghost_bn", type=int, default=0,
                    help="virtual BN sub-batch (0 = off): run a large "
                         "physical batch with the accumulation recipe's "
                         "micro-batch BN statistics")
    ap.add_argument("--count_weight", type=float, default=0.0,
                    help="> 0 enables the per-class count head and "
                         "weights its smooth-L1 on log1p counts")
    ap.add_argument("--image_trunk", default="resnet18",
                    choices=("resnet18", "resnet50"),
                    help="image-trunk backbone (the label trunk stays "
                         "resnet18)")
    ap.add_argument("--cache_images", action="store_true",
                    help="cache decoded+resized images in worker RAM "
                         "across epochs (~0.5-1 MB per image per worker)")
    ap.add_argument("--severity_mix", default="balanced")
    ap.add_argument("--no_augment", action="store_true")
    ap.add_argument("--aug_prob", type=float, default=0.5)
    ap.add_argument("--val_split", type=float, default=-1.0,
                    help="< 0 uses the dataset's own discipline")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output directory (default artifacts_<QA_DATASET>)")
    args = ap.parse_args()
    if args.out is None:
        args.out = "artifacts_" + os.environ.get("QA_DATASET", "voc")
    if args.ghost_bn > 0 and args.batch_size % args.ghost_bn:
        ap.error(f"--batch_size {args.batch_size} must be a multiple of "
                 f"--ghost_bn {args.ghost_bn}; GhostBatchNorm would otherwise "
                 f"fall back to whole-batch statistics")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.p_exact >= 0 or args.p_edge > 0:
        # Rebinds the module attribute: the dataset's positive draw and
        # every corrupt variant's base draw resolve good_label at call
        # time; recipe_state / worker_init carry it into the workers.
        qa_data.good_label = functools.partial(
            qa_data.good_label,
            p_exact=max(0.0, args.p_exact), p_edge=args.p_edge)
        print(f"positive sampler p_exact set to {args.p_exact:.2f}, "
              f"p_edge {args.p_edge:.2f}", flush=True)

    pool = ds_conf.load_pool()
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    if args.limit:
        pool = pool[:args.limit]
    val_split = args.val_split if args.val_split >= 0 else ds_conf.VAL_SPLIT
    n_val = max(10, int(len(pool) * val_split))
    val_pool, train_pool = pool[:n_val], pool[n_val:]
    if not train_pool:
        sys.exit(f"--limit {args.limit} leaves no training images: "
                 f"validation keeps at least 10")
    # Bank and co-occurrence come from the training images only, so the
    # validation negatives never see their own boxes in the bank.
    qa_data.build_context(train_pool)
    per_box = args.per_box_weight > 0 or args.margin_weight > 0
    print(f"{len(train_pool)} train / {len(val_pool)} validation images; "
          f"unified mixer detector={args.detector_share} "
          f"ih={args.ih_share} compose={args.compose} "
          f"swap={args.swap_share} severity_mix={args.severity_mix}; "
          f"augment={not args.no_augment}; per_box={per_box} "
          f"(support {args.per_box_weight}, margin {args.margin_weight}); "
          f"count_weight={args.count_weight}",
          flush=True)

    def parity(pool_part):
        paths = []
        for img, packed, _, _ in pool_part:
            paths += [(img, packed, 1), (img, packed, 0)]
        return paths

    grid = ds_conf.FUSION_RES if args.aux_weight > 0 else 0
    train_ds = qa_data.Data(parity(train_pool),
                            augment=not args.no_augment,
                            aug_prob=args.aug_prob,
                            detector_share=args.detector_share,
                            ih_share=args.ih_share, compose=args.compose,
                            swap_share=args.swap_share,
                            severity_mix=args.severity_mix,
                            deterministic=True, seed=args.seed,
                            gt_map_grid=grid, sparse_planes=True,
                            per_box=per_box,
                            gt_counts=args.count_weight > 0,
                            cache_images=args.cache_images)
    val_ds = qa_data.Data(parity(val_pool), augment=False,
                          detector_share=args.detector_share,
                          ih_share=args.ih_share, compose=args.compose,
                          swap_share=args.swap_share,
                          severity_mix=args.severity_mix,
                          deterministic=True, seed=args.seed + 1,
                          sparse_planes=True, per_box=per_box)
    train_ds.recipe = val_ds.recipe = recipe_state()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              drop_last=True,
                              prefetch_factor=(2 if args.workers > 0 else None),
                              collate_fn=qa_data.collate_sparse,
                              worker_init_fn=worker_init,
                              persistent_workers=args.workers > 0,
                              pin_memory=True)
    # Validation runs interleaved with training-reserved GPU memory, so
    # cap its batch: at large training batches the unpacked val planes
    # would not fit beside the training allocator's pools.
    val_loader = DataLoader(val_ds, batch_size=min(args.batch_size, 32),
                            num_workers=args.workers,
                            prefetch_factor=(1 if args.workers > 0 else None),
                            collate_fn=qa_data.collate_sparse,
                            worker_init_fn=worker_init,
                            persistent_workers=args.workers > 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Fixed input shapes every step: let cuDNN autotune kernels.
        torch.backends.cudnn.benchmark = True
    use_amp = args.amp and device.type == "cuda"
    model = FusionNet(freeze_through=args.freeze_through, per_box=per_box,
                      count_head=args.count_weight > 0,
                      image_arch=args.image_trunk, device=device)
    if args.ghost_bn > 0:
        from qa_model import ghostify
        ghostify(model, args.ghost_bn)
        model.to(device)
        print(f"ghost BatchNorm: virtual sub-batch {args.ghost_bn}",
              flush=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{trainable / 1e6:.1f}M trainable parameters, "
          f"{ds_conf.NUM_CLASSES} classes, input {ds_conf.RES}px, fusion "
          f"{ds_conf.FUSION_RES}x{ds_conf.FUSION_RES}, frozen through "
          f"{args.freeze_through}, amp={use_amp}", flush=True)

    opt = torch.optim.AdamW(
        model.param_groups(args.lr, args.finetune_mult),
        betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    opt_steps = max(1, (len(train_loader) + args.accum - 1) // args.accum)
    warmup = args.warmup_epochs * opt_steps
    total_steps = args.epochs * opt_steps

    def lr_at(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"effective batch {args.batch_size * args.accum} "
          f"({args.batch_size} x {args.accum} accumulated)", flush=True)

    ema = None
    if args.ema_decay > 0:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def ema_update():
        if ema is None:
            return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(args.ema_decay)
                    ema[k].add_(v, alpha=1 - args.ema_decay)
                else:
                    ema[k].copy_(v)

    os.makedirs(args.out, exist_ok=True)
    best, best_ema, history, start_epoch = -1.0, -1.0, [], 0
    ckpt_path = os.path.join(args.out, "checkpoint.pth")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best = ckpt["best"]
        best_ema = ckpt["best_ema"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"]
        if args.ema_decay > 0:
            if ckpt.get("ema") is not None:
                ema = {k: v.to(device) for k, v in ckpt["ema"].items()}
            else:  # resumed from an EMA-less run: start the average here
                ema = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        if device.type == "cuda" and ckpt["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        print(f"resumed from {ckpt_path} at epoch {start_epoch + 1}",
              flush=True)

    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        running, steps = 0.0, 0
        opt.zero_grad()
        for i, batch in enumerate(train_loader):
            img, planes, y = batch[0], batch[1], batch[3]
            img, planes = qa_data.to_dense_batch(img, planes, batch[2],
                                                 device)
            use_aux = args.aux_weight > 0 and model.class_corr
            use_cnt = args.count_weight > 0 and model.count_head
            boxes_t = batch[-1].to(device) if per_box else None
            with torch.amp.autocast("cuda", enabled=use_amp):
                if use_aux or per_box or use_cnt:
                    logits, cmap, counts, box_out = model(
                        planes, img,
                        boxes=boxes_t[:, :6] if per_box else None,
                        return_aux=True)
                    loss = crit(logits, y.long().to(device))
                    if use_aux:
                        loss = loss + args.aux_weight * \
                            nn.functional.cross_entropy(
                                cmap, batch[4].long().to(device),
                                ignore_index=255)
                    if use_cnt:
                        # counts ride just before the box rows; the last
                        # element flags label-side counts that describe
                        # the shipped image (0 under an image swap)
                        cnt_t = batch[-2 if per_box else -1].to(device)
                        valid = cnt_t[:, -1] > 0
                        if bool(valid.any()):
                            # log1p scale: missing one of one object
                            # matters more than one of thirty
                            loss = loss + args.count_weight * \
                                nn.functional.smooth_l1_loss(
                                    torch.log1p(counts.float()[valid]),
                                    torch.log1p(cnt_t[valid, :-1]))
                    if per_box and box_out is not None and box_out.shape[0]:
                        if args.per_box_weight > 0:
                            loss = loss + args.per_box_weight * \
                                nn.functional. \
                                binary_cross_entropy_with_logits(
                                    box_out[:, 0].float(), boxes_t[:, 6])
                        mval = boxes_t[:, 8] > 0
                        if args.margin_weight > 0 and bool(mval.any()):
                            loss = loss + args.margin_weight * \
                                nn.functional.smooth_l1_loss(
                                    box_out[:, 1].float()[mval],
                                    boxes_t[:, 7][mval])
                    loss = loss / args.accum
                else:
                    logits = model(planes, img)
                    loss = crit(logits, y.long().to(device)) / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()
                ema_update()
            running += loss.item() * args.accum
            steps += 1
        if steps % args.accum:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            scheduler.step()
            ema_update()
        auc, acc = evaluate_loader(model, val_loader, device, use_amp)
        star = ""
        if auc > best:
            best = auc
            torch.save(model.state_dict(),
                       os.path.join(args.out, "best_model.pth"))
            star = "  *best*"
        ema_note = ""
        if ema is not None:
            raw_state = {k: v.detach().clone()
                         for k, v in model.state_dict().items()}
            model.load_state_dict(ema)
            e_auc, e_acc = evaluate_loader(model, val_loader, device,
                                           use_amp)
            if e_auc > best_ema:
                best_ema = e_auc
                torch.save(model.state_dict(),
                           os.path.join(args.out, "best_ema.pth"))
                ema_note = " *ema-best*"
            ema_note = f" ema={e_auc:.3f}{ema_note}"
            model.load_state_dict(raw_state)
        entry = {"epoch": epoch + 1, "loss": running / max(1, steps),
                 "val_auc": auc, "val_acc": acc,
                 "lr": scheduler.get_last_lr()[0]}
        if ema is not None:
            entry.update({"ema_val_auc": e_auc, "ema_val_acc": e_acc})
        history.append(entry)
        with open(os.path.join(args.out, "training_log.json"), "w") as f:
            json.dump({"args": vars(args), "best_val_auc": best,
                       "best_ema_val_auc": best_ema, "history": history},
                      f, indent=2)
            f.write("\n")
        torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": opt.state_dict(), "ema": ema,
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "best": best,
                    "best_ema": best_ema, "history": history,
                    "rng": {"python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": (torch.cuda.get_rng_state_all()
                                     if device.type == "cuda" else None)}},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        print(f"epoch {epoch + 1}/{args.epochs} "
              f"loss={running / max(1, steps):.4f} "
              f"val_auc={auc:.3f} val_acc={acc:.3f}{star}{ema_note}",
              flush=True)

    print(f"best raw AUC {best:.3f}, best EMA AUC {best_ema:.3f}; "
          f"artifacts in {args.out}/")


if __name__ == "__main__":
    main()
