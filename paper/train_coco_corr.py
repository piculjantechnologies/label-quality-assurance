"""Train CorrNet (coco_corrnet.py) on the paper's COCO label-quality task.

run.sh carries the released recipe; the bare defaults here are not it
(they keep the paper's Adam, and research options). What run.sh's flags
select, by README item:

  candidates  --p_exact 0.5 (item 3); --swap_share 0.1 (item 11);
              --detector_share 0.1 (item 12); --hard_negatives
              --paper_mix 0.5 (item 6: the remaining negatives half from
              the paper's Section 5 generator, half from
              coco_hard_negatives); --augment_p 0.1 (the thesis's
              Listing 8.11 set, data_loader.build_augment).
  network     --corr_masked --corr_max --corr_cos (item 4);
              --pretrained_image --input_norm --freeze_image_bn
              --img_lr_mult 0.1 (item 5); --raster_head --box_weight 0.5
              --cls_weight 0.5 (item 9).
  training    --optimizer adamw --lr 1e-3 --weight_decay 1e-2
              --warmup_epochs 2 --cosine; --epochs 160 --patience 0;
              --batch_size 32 --accum 2 --ghost_bn 8 (item 8); --amp;
              --ema_decay 0.998 (item 7).
  validation  --val_split 0.1 --val_rounds 4 --val_test_protocol: a fixed
              set drawn as evaluate.py draws its test pool; best_model.pth
              is the epoch with the highest validation ROC-AUC.

The loss is two-class cross-entropy on the verdict's logits
(interpretation 1). Importing coco_a3_plausible installs its generator
(A3 swapping to a context-plausible absent class) as data_loader.corrupt,
which trains the negatives when --hard_negatives is off.
--raster_head adds the per-cell verification head (coco_corrnet.py),
trained with two per-cell losses next to the verdict's cross-entropy:
the claimed cells' support against the per-box form of Equations 1-4
(--box_weight) and every cell's class against the ground truth
(--cls_weight), both from data_loader.cell_targets.

The generator's configuration (co-occurrence tables, sampler patches, box
bank) is carried on the dataset objects and installed in every DataLoader
worker by worker_init, so the loaders draw the same samples under every
multiprocessing start method (fork, spawn, forkserver).

Example (bare defaults; ./run.sh runs the released recipe):
    python train_coco_corr.py \
        --images /path/to/coco/val2017 \
        --annotations /path/to/coco/annotations/instances_val2017.json
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
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import coco_a3_plausible as a3          # patches data_loader.corrupt
import data_loader as dl
from coco_corrnet import CorrNet, ghostify
from data_loader import LabelQualityDataset, load_coco


def recipe_state(a4=None):
    """Snapshot of the generator's process-level configuration, taken once
    main() has applied it: the a3 co-occurrence tables and A2 offset
    floor, the positive sampler's keyword overrides (dl.good), the active
    corrupt function (dl.corrupt) and, with --hard_negatives (a4 is the
    coco_hard_negatives module), the box bank and its subtype shares.

    The datasets carry this dict so that DataLoader workers started from
    a fresh interpreter -- the spawn and forkserver start methods: macOS,
    and Linux from Python 3.14 -- are brought to the parent's state by
    worker_init instead of running the bare data_loader generator (where
    a3's tables are None and pmi raises). Everything here pickles: plain
    tables and module-level functions; dl.good's partial is stored as its
    keywords because a partial over a rebound module attribute does not.
    Fork-started workers inherit the state and re-apply the same values.
    """
    return {
        "cooc": (a3._cooc, a3._freq, a3._total),
        "a2_min_offset": a3.A2_MIN_OFFSET,
        "good_kw": (dict(dl.good.keywords)
                    if isinstance(dl.good, functools.partial) else None),
        "corrupt": dl.corrupt,
        "bank": ((a4._bank, a4.A2_SHARE, a4.B1_SHARE)
                 if a4 is not None else None),
        "paper_mix": a4.PAPER_MIX if a4 is not None else 0.0,
    }


def install_recipe(state):
    """Apply a recipe_state() dict to this process's generator modules."""
    if state is None:
        return
    a3._cooc, a3._freq, a3._total = state["cooc"]
    a3.A2_MIN_OFFSET = state["a2_min_offset"]
    good = dl.good.func if isinstance(dl.good, functools.partial) else dl.good
    dl.good = (functools.partial(good, **state["good_kw"])
               if state["good_kw"] else good)
    dl.corrupt = state["corrupt"]
    if state["bank"] is not None:
        import coco_hard_negatives as a4
        a4._bank, a4.A2_SHARE, a4.B1_SHARE = state["bank"]
        a4.PAPER_MIX = state["paper_mix"]


def worker_init(worker_id):
    """DataLoader worker_init_fn: one cv2 thread per worker, then the
    generator recipe carried by the worker's dataset (recipe_state)."""
    dl.cv2_single_thread_worker(worker_id)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        install_recipe(getattr(info.dataset, "recipe", None))


class ValRounds(torch.utils.data.Dataset):
    """Both candidates of every validation image over draw rounds
    0 .. rounds - 1 (the base dataset's epochs): a fixed set, the same
    every epoch. refit.py draws from round 4 onward."""

    def __init__(self, base, rounds):
        self.base, self.rounds = base, rounds
        self.recipe = base.recipe

    def __len__(self):
        return len(self.base) * self.rounds

    def __getitem__(self, i):
        r, j = divmod(i, len(self.base))
        self.base.set_epoch(r)
        return self.base[j]


def test_protocol_recipe(state):
    """recipe_state() for validation drawn as evaluate.py draws its test
    pool: the paper's Section 5 generator, good candidates inside the
    uncertainty regions only (p_exact 0)."""
    return dict(state, good_kw={"p_exact": 0.0},
                corrupt=dl.PAPER_CORRUPT, bank=None, paper_mix=0.0)


def evaluate_loader(model, loader, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for cats, background, y in loader:
            if isinstance(cats, (tuple, list)):
                cats = dl.planes_to_dense(cats, background, device)
            logits = model(cats.to(device), background.to(device))
            scores += torch.softmax(logits, 1)[:, 1].tolist()
            labels += y.tolist()
    model.train()
    auc = roc_auc_score(labels, scores)
    acc = float(np.mean((np.array(scores) >= 0.5) == np.array(labels)))
    return auc, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=125)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--warm", default="",
                    help="paper-net checkpoint to adopt (fresh optimizer)")
    ap.add_argument("--init", default="",
                    help="CorrNet state dict to start from (fine-tune)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>/checkpoint.pth if present")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", choices=("adam", "adamw"), default="adam",
                    help="adam: the paper's (zero weight decay); adamw: "
                         "decoupled --weight_decay")
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="AdamW weight decay (--optimizer adamw)")
    ap.add_argument("--warmup_epochs", type=int, default=0,
                    help="linear learning-rate warmup, in epochs")
    ap.add_argument("--cosine", action="store_true",
                    help="cosine learning-rate decay to 0 over --epochs "
                         "(after the warmup); off = constant rate")
    ap.add_argument("--input_norm", action="store_true",
                    help="normalise the image with the ImageNet statistics "
                         "inside the model instead of folding them into "
                         "the pretrained conv1")
    ap.add_argument("--freeze_image_bn", action="store_true",
                    help="keep the image trunk's BatchNorm statistics at "
                         "their pretrained values (eval mode in training)")
    ap.add_argument("--augment_p", type=float, default=0.0,
                    help="share of training samples augmented with "
                         "data_loader.build_augment()")
    ap.add_argument("--detector_share", type=float, default=0.0,
                    help="share of negatives taken from a detector, "
                         "labelled by the whole-label metric")
    ap.add_argument("--swap_share", type=float, default=0.0,
                    help="share of negatives pairing a good label with "
                         "another training image")
    ap.add_argument("--val_rounds", type=int, default=1,
                    help="draw rounds of the fixed validation set")
    ap.add_argument("--val_test_protocol", action="store_true",
                    help="draw validation as evaluate.py draws the test "
                         "pool (paper generator, p_exact 0) instead of "
                         "with the training recipe")
    ap.add_argument("--ema_decay", type=float, default=0.0,
                    help="per-step weight EMA; 0 disables. When active, "
                         "validation and saved models use the EMA weights")
    ap.add_argument("--hard_negatives", action="store_true",
                    help="image-hard negatives via coco_hard_negatives")
    ap.add_argument("--corr_masked", action="store_true",
                    help="add box-masked correlation pooling (512-d)")
    ap.add_argument("--corr_max", action="store_true",
                    help="add max-pooled correlation channel (512-d)")
    ap.add_argument("--corr_cos", action="store_true",
                    help="added corr blocks use L2-normalised features")
    ap.add_argument("--corr16", action="store_true",
                    help="box-masked correlation at stride 16 (256-d)")
    ap.add_argument("--corr8", action="store_true",
                    help="box-masked correlation at stride 8 (128-d)")
    ap.add_argument("--corr4", action="store_true",
                    help="box-masked correlation at stride 4 (64-d)")
    ap.add_argument("--pretrained_image", action="store_true",
                    help="ImageNet init for the image trunk (init only, "
                         "same module layout); with --init, the source's "
                         "model_1.* keys are skipped so the ImageNet "
                         "features survive the warm start")
    ap.add_argument("--raster_head", action="store_true",
                    help="add the per-cell verification head that reads "
                         "the label raster (needs --sparse_planes)")
    ap.add_argument("--box_weight", type=float, default=0.0,
                    help="weight of --raster_head's per-cell support loss "
                         "against the per-box form of Equations 1-4")
    ap.add_argument("--cls_weight", type=float, default=0.0,
                    help="weight of --raster_head's per-cell class loss "
                         "against the ground-truth class map")
    ap.add_argument("--hflip", action="store_true",
                    help="horizontally flip training samples (image, label "
                         "raster and cell targets together)")
    ap.add_argument("--paper_mix", type=float, default=0.0,
                    help="with --hard_negatives, the share of negatives "
                         "drawn by the paper's own Section 5 generator")
    ap.add_argument("--img_lr_mult", type=float, default=1.0,
                    help="learning-rate factor for the image trunk")
    ap.add_argument("--a2_min_offset", type=float, default=0.0,
                    help="floor of A2's offset draw as a fraction of the box "
                         "dimension (0 keeps the paper's 0.02, which leaves "
                         "the classes sub-pixel apart for small boxes)")
    ap.add_argument("--a2_share", type=float, default=0.0,
                    help="fraction of negatives forced to A2, the only "
                         "purely geometric error (0 keeps the generator's "
                         "own mixture, which leaves A2 at 1/8)")
    ap.add_argument("--p_exact", type=float, default=-1.0,
                    help="probability that a positive is the ground truth "
                         "verbatim (negative keeps data_loader's default; "
                         "region-jittered positives sit as close as 0.02 x "
                         "box dim to an A2 negative, sub-pixel for small "
                         "boxes, so crisp positives are what makes the "
                         "outward half of A2 offenses separable)")
    ap.add_argument("--sparse_planes", action="store_true",
                    help="ship only non-empty class planes from the "
                         "workers and scatter on the GPU (transport "
                         "only; batches are bit-identical)")
    ap.add_argument("--ghost_bn", type=int, default=0,
                    help="virtual BN sub-batch (0 = off): run a large "
                         "physical batch with the accumulation recipe's "
                         "micro-batch BN statistics")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_split", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=0,
                    help="also keep model_eNNN.pth every N epochs")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()
    if args.raster_head and not args.sparse_planes:
        ap.error("--raster_head needs --sparse_planes")
    if args.paper_mix > 0 and not args.hard_negatives:
        ap.error("--paper_mix mixes into --hard_negatives")
    if args.ghost_bn > 0 and args.batch_size % args.ghost_bn:
        ap.error(f"--batch_size {args.batch_size} must be a multiple of "
                 f"--ghost_bn {args.ghost_bn}; GhostBatchNorm would otherwise "
                 f"fall back to whole-batch statistics")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    a3.build_cooccurrence(os.path.expanduser(args.annotations))
    if args.a2_min_offset > 0:
        a3.A2_MIN_OFFSET = args.a2_min_offset
        print(f"A2 offset floor raised to {args.a2_min_offset:.2f} of the "
              f"box dimension (paper: 0.02)", flush=True)
    if args.p_exact >= 0:
        # Rebinds the module attribute: the dataset's positive draw and
        # every corrupt variant's base draw resolve dl.good at call time.
        dl.good = functools.partial(dl.good, p_exact=args.p_exact)
        print(f"positive sampler p_exact set to {args.p_exact:.2f} "
              f"(data_loader default {dl.P_EXACT:.2f})", flush=True)
    a4 = None
    if args.hard_negatives:
        import coco_hard_negatives as a4
        a4.build_box_bank(os.path.expanduser(args.annotations))
        print("image-hard negatives active (A1-one/A2/A3/R + bank B1/B2)",
              flush=True)
        if args.a2_share > 0:
            a4.A2_SHARE = args.a2_share
            print(f"A2 (geometric) share of negatives forced to "
                  f"{args.a2_share:.2f}", flush=True)
        if args.paper_mix > 0:
            a4.PAPER_MIX = args.paper_mix
            dl.corrupt = a4.corrupt_mixed
            print(f"{args.paper_mix:.2f} of negatives from the paper's "
                  f"Section 5 generator", flush=True)
    samples, class_names = load_coco(os.path.expanduser(args.annotations))
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    if args.limit:
        samples = samples[:args.limit]
    n_val = max(10, int(len(samples) * args.val_split))
    val_samples, train_samples = samples[:n_val], samples[n_val:]
    if not train_samples:
        sys.exit(f"--limit {args.limit} leaves no training images: "
                 f"validation keeps at least 10")
    print(f"{len(train_samples)} train / {len(val_samples)} validation "
          f"images", flush=True)

    images = os.path.expanduser(args.images)
    train_ds = LabelQualityDataset(train_samples, images, args.size,
                                   seed=args.seed,
                                   sparse_planes=args.sparse_planes,
                                   hflip=args.hflip,
                                   cell_targets=args.raster_head,
                                   augment_p=args.augment_p,
                                   detector_share=args.detector_share,
                                   swap_share=args.swap_share,
                                   class_names=class_names)
    val_ds = LabelQualityDataset(val_samples, images, args.size,
                                 seed=args.seed + 1,
                                 sparse_planes=args.sparse_planes)
    train_ds.recipe = val_ds.recipe = recipe_state(a4)
    if args.val_test_protocol:
        val_ds.recipe = test_protocol_recipe(val_ds.recipe)
    if args.val_rounds > 1 or args.val_test_protocol:
        val_ds = ValRounds(val_ds, args.val_rounds)
    print(f"validation: {len(val_ds)} candidates ({args.val_rounds} "
          f"round(s), {'test protocol' if args.val_test_protocol else 'training recipe'})",
          flush=True)
    collate = dl.collate_sparse_planes if args.sparse_planes else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              drop_last=True, prefetch_factor=(1 if args.workers > 0 else None),
                              collate_fn=(dl.collate_cells
                                          if args.raster_head else collate),
                              worker_init_fn=worker_init)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.workers, prefetch_factor=(1 if args.workers > 0 else None),
                            collate_fn=collate,
                            worker_init_fn=worker_init)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    model = CorrNet(device, corr_masked=args.corr_masked,
                    corr_max=args.corr_max, corr_cos=args.corr_cos,
                    corr16=args.corr16, corr8=args.corr8, corr4=args.corr4,
                    pretrained_img=args.pretrained_image,
                    raster_head=args.raster_head,
                    input_norm=args.input_norm,
                    freeze_image_bn=args.freeze_image_bn)
    if args.ghost_bn > 0:
        ghostify(model, args.ghost_bn)
        model.to(device)
        print(f"ghost BN: virtual sub-batch {args.ghost_bn}", flush=True)
    if args.pretrained_image:
        print("image trunk initialized from ImageNet weights", flush=True)
    if args.warm:
        model.warm_start(os.path.expanduser(args.warm))
        print(f"warm-started from {args.warm}", flush=True)
    if args.init:
        src = torch.load(os.path.expanduser(args.init), map_location="cpu",
                         weights_only=False)
        if args.pretrained_image:
            src = {k: v for k, v in src.items()
                   if not k.startswith("model_1.")}
            print("hybrid warm start: source image trunk skipped",
                  flush=True)
        if src["fc.1.weight"].shape != model.fc[1].weight.shape:
            model.adopt_corr(src)
            print(f"adopted (columns widened) from {args.init}", flush=True)
        else:
            own = model.state_dict()
            for k, v in src.items():
                if k != "corr_cfg":     # this model records its own
                    own[k].copy_(v)
            model.load_state_dict(own)
            print(f"initialized from {args.init}", flush=True)
    # _fold_input_convention rescales a pretrained conv1's weights by
    # 1/(255*std) ~= 1/57.6. Adam's step size is fixed in parameter units,
    # so on the shrunken weights the same lr means ~57x larger RELATIVE
    # steps. Scaling conv1's lr by the same factor restores exactly the
    # trajectory the unfolded parameterisation would have had (Adam
    # normalises gradient magnitude away, so lr is the only knob).
    # --img_lr_mult scales the whole image trunk's lr, conv1's included.
    # With --input_norm (the released recipe, README item 5) nothing is
    # folded and conv1 trains with the rest of the trunk.
    fold = 255.0 * 0.226                # mean ImageNet std; per-channel
    lr_img = args.lr * args.img_lr_mult                 # spread is <3%
    conv1 = (list(model.model_1[0].parameters())
             if args.pretrained_image and not args.input_norm else [])
    trunk = ([p for p in model.model_1.parameters()
              if all(p is not q for q in conv1)]
             if args.img_lr_mult != 1.0 else [])
    ids = {id(p) for p in conv1 + trunk}
    groups = [{"params": [p for p in model.parameters()
                          if id(p) not in ids]}]
    if trunk:
        groups.append({"params": trunk, "lr": lr_img})
        print(f"image trunk lr {lr_img:.2e}", flush=True)
    if conv1:
        groups.append({"params": conv1, "lr": lr_img / fold})
        print(f"conv1 lr scaled to {lr_img / fold:.2e} to match the "
              f"folded weight scale", flush=True)
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999),
                                eps=1e-8, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.Adam(groups, lr=args.lr)
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"effective batch {args.batch_size * args.accum} "
          f"({args.batch_size} x {args.accum} accumulated), amp={use_amp}",
          flush=True)
    # learning-rate schedule per optimizer step: linear warmup, then cosine
    # decay to 0 at the last step (or constant)
    steps_per_epoch = -(-len(train_loader) // args.accum)
    warmup = args.warmup_epochs * steps_per_epoch
    total = args.epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        if not args.cosine:
            return 1.0
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    print(f"{args.optimizer}, lr {args.lr:g}, weight decay "
          f"{args.weight_decay:g}, warmup {args.warmup_epochs} epoch(s), "
          f"{'cosine' if args.cosine else 'constant'} over "
          f"{steps_per_epoch} steps/epoch", flush=True)

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
    best, best_epoch, history, start_epoch = -1.0, 0, [], 0
    ckpt_path = os.path.join(args.out, "checkpoint.pth")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("sched") is not None:
            sched.load_state_dict(ckpt["sched"])
        best = ckpt["best"]
        best_epoch = ckpt["best_epoch"]
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
            cats, background, y = batch[:3]
            pack = cats
            if isinstance(cats, (tuple, list)):
                cats = dl.planes_to_dense(cats, background, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if args.raster_head and (args.box_weight > 0
                                         or args.cls_weight > 0):
                    logits, sup, cls_map = model(cats.to(device),
                                                 background.to(device),
                                                 aux=True)
                    loss = crit(logits, y.long().to(device))
                    if args.box_weight > 0:
                        # support of each claimed (class plane, cell)
                        # against that cell's box_good (255 = unclaimed)
                        (si, ci), _ = pack
                        T = batch[3].to(device)
                        claimed = T != 255
                        s = sup[si.to(device), ci.to(device)][claimed]
                        if s.numel():
                            loss = loss + args.box_weight * \
                                F.binary_cross_entropy_with_logits(
                                    s, T[claimed].float())
                    if args.cls_weight > 0:
                        loss = loss + args.cls_weight * F.cross_entropy(
                            cls_map.float(), batch[4].long().to(device))
                else:
                    logits = model(cats.to(device), background.to(device))
                    loss = crit(logits, y.long().to(device))
                loss = loss / args.accum
            scaler.scale(loss).backward()
            if (i + 1) % args.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
                ema_update()
            running += loss.item() * args.accum
            steps += 1
        if steps % args.accum:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()
            ema_update()
        raw_state = None
        if ema is not None:
            raw_state = {k: v.detach().clone()
                         for k, v in model.state_dict().items()}
            model.load_state_dict(ema)
        auc, acc = evaluate_loader(model, val_loader, device)
        star = ""
        if auc > best:
            best, best_epoch = auc, epoch
            torch.save(model.state_dict(),
                       os.path.join(args.out, "best_model.pth"))
            star = "  *best*"
        torch.save(model.state_dict(), os.path.join(args.out, "model.pth"))
        if args.save_every and (epoch + 1) % args.save_every == 0:
            torch.save(model.state_dict(), os.path.join(
                args.out, f"model_e{epoch + 1:03d}.pth"))
        if raw_state is not None:
            model.load_state_dict(raw_state)
        history.append({"epoch": epoch + 1, "loss": running / max(1, steps),
                        "val_auc": auc, "val_acc": acc})
        with open(os.path.join(args.out, "training_log.json"), "w") as f:
            json.dump({"args": vars(args), "best_val_auc": best,
                       "history": history}, f, indent=2)
            f.write("\n")
        torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": opt.state_dict(), "ema": ema,
                    "scaler": scaler.state_dict(), "best": best,
                    "sched": sched.state_dict(),
                    "best_epoch": best_epoch, "history": history,
                    "rng": {"python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state(),
                            "cuda": (torch.cuda.get_rng_state_all()
                                     if device.type == "cuda" else None)}},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        print(f"epoch {epoch + 1}/{args.epochs} "
              f"loss={running / max(1, steps):.4f} "
              f"val_auc={auc:.3f} val_acc={acc:.3f}{star}", flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            print(f"early stop: no validation AUC improvement in "
                  f"{args.patience} epochs")
            break

    print(f"best validation AUC {best:.3f}; artifacts in {args.out}/")


if __name__ == "__main__":
    main()
