"""Held-out refit of a raster-head model's verdict layer, folded into the
checkpoint (the last step of run.sh; README item 10).

The verdict's good-minus-bad margin is glob + box_mix([smin, mean,
log1p(count)]) + bias: glob is the fully connected layers' margin, and the
three box_mix inputs are the raster head's smooth minimum, mean and count
over the claimed cells. Training fits these weights on the training
images, which the model has partly memorised, and on the training
candidate mix (exact ground truth, image-hard, image-swap and detector
candidates), which differs from the test protocol's; on unseen
test-protocol draws the verdict condemns too many good labels. This
script refits glob's weight, the three box_mix weights and the intercept
by logistic regression on held-out data -- the training run's held-out
val2017 images (its --seed and --val_split), the paper's Section 5
generator, good candidates drawn inside the uncertainty regions (p_exact
0), draw rounds --round_offset onward (validation during training uses
rounds 0-3) -- and folds the fit in: the final fc layer is scaled by glob's
coefficient, box_mix takes the three box weights with bias 0, and the
intercept goes into the final bias as +b/2 on the good row and -b/2 on the
bad row. P(good) >= 0.5 is then the fit's zero point. The network and its
inputs are unchanged; only these five numbers move, and ROC-AUC can move
with them because the weighting of glob against the head changes.

With run.sh's --val_split 0.1 --seed 0 and the default --round_offset 4
--rounds 8, the population is the 454 held-out images over draw rounds
4-11: 7264 candidates, both of every image in every round.

The report also cross-fits: the images are split into two halves and each
half is scored by a fit on the other, so the pooled accuracy, ROC-AUC and
per-subtype recall are measured on images the fit did not see; they sit
next to the uncalibrated argmax and a cross-fitted threshold-only fit on
the same population, each with its Section 5-weighted accuracy (the
per-subtype recalls weighted by Section 5's subtype probabilities). The
fold check re-scores the first two rounds with the folded checkpoint.
The train2017 test pool is never touched.

    python3 refit.py artifacts/best_model.pth \\
        --out artifacts/best_model_refit.pth --report artifacts/refit.json \\
        --images /path/to/coco/val2017 \\
        --annotations /path/to/coco/annotations/instances_val2017.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import data_loader as dl
import evaluate as ev              # records each bad draw's subtype
from coco_corrnet import CorrNet

FEATURES = ("glob", "smin", "mean", "log1p_count")
WEIGHTS = {"A1": 1 / 6, "A2": 1 / 6, "A3": 1 / 6, "B1": 1 / 4, "B2": 1 / 4}


class HeldOut(Dataset):
    """Both candidates of every held-out image over draw rounds offset ..
    offset + rounds - 1, each with its subtype and image index."""

    p_exact = 0.0                       # read by evaluate.worker_init

    def __init__(self, base, offset, rounds, subtypes, recorded):
        self.base, self.offset, self.rounds = base, offset, rounds
        self.subtypes, self.recorded = subtypes, recorded

    def __len__(self):
        return len(self.base) * self.rounds

    def __getitem__(self, i):
        r, j = divmod(i, len(self.base))
        self.base.set_epoch(self.offset + r)
        item = self.base[j]
        st = "good" if j % 2 == 0 else self.recorded[0]
        return (*item, self.subtypes.index(st), j // 2)


def score(model, loader, device):
    """Per candidate: the model's own margin, the verdict inputs (glob and
    what box_mix reads), label, subtype and image index -- fp32, no
    autocast."""
    seen = {}
    hooks = [model.fc.register_forward_hook(
                 lambda m, i, o: seen.__setitem__("fc", o)),
             model.box_mix.register_forward_pre_hook(
                 lambda m, i: seen.__setitem__("box", i[0]))]
    rec = {k: [] for k in ("margin", "X", "label", "subtype", "image")}
    with torch.no_grad():
        for batch in loader:
            cats, background, y = batch[:3]
            seen.clear()
            out = model(dl.planes_to_dense(cats, background, device),
                        background.to(device)).float()
            g = seen["fc"].float()
            box = seen["box"].float()
            rec["margin"] += (out[:, 1] - out[:, 0]).tolist()
            rec["X"] += torch.cat([(g[:, 1] - g[:, 0])[:, None], box],
                                  1).tolist()
            rec["label"] += y.tolist()
            rec["subtype"] += batch[-2].tolist()
            rec["image"] += batch[-1].tolist()
    for h in hooks:
        h.remove()
    return {k: np.array(v) for k, v in rec.items()}


def summary(s, y, st, names, thr=0.0):
    """Accuracy, good recall, per-subtype recall and the Section 5
    weighted accuracy of scores s at threshold thr."""
    pg = s >= thr
    out = {"accuracy": float(np.mean(pg == y)),
           "good_recall": float(pg[y == 1].mean())}
    for k, name in enumerate(names[1:], 1):
        out[name] = float((~pg[st == k]).mean())
    out["section5_accuracy"] = (out["good_recall"] + sum(
        w * out[n] for n, w in WEIGHTS.items())) / 2
    return out


def best_threshold(s, y):
    """The accuracy-optimal threshold on scores s."""
    o = np.sort(s)
    cands = np.concatenate([[o[0] - 1], (o[:-1] + o[1:]) / 2, [o[-1] + 1]])
    return max((float(np.mean((s >= t) == y)), t) for t in cands)[1]


def fit(X, y, C):
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--images", required=True,
                    help="COCO val2017 image directory (the training split)")
    ap.add_argument("--annotations", required=True,
                    help="COCO instances_val2017.json")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0,
                    help="the training run's --seed")
    ap.add_argument("--val_split", type=float, default=0.1,
                    help="the training run's --val_split")
    ap.add_argument("--round_offset", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorrNet.from_checkpoint(args.checkpoint, device).eval()
    if not model.raster_head:
        sys.exit("refit.py needs a raster_head checkpoint")

    samples, _ = dl.load_coco(os.path.expanduser(args.annotations))
    random.Random(args.seed).shuffle(samples)   # as train_coco_corr.py does
    val = samples[:max(10, int(len(samples) * args.val_split))]
    base = dl.LabelQualityDataset(val, os.path.expanduser(args.images),
                                  args.size, seed=args.seed + 1,
                                  sparse_planes=True)
    ev.install_p_exact(0.0)

    def loader(offset, rounds):
        return DataLoader(HeldOut(base, offset, rounds, ev.SUBTYPES,
                                  ev._last_subtype),
                          batch_size=args.batch_size,
                          num_workers=args.workers,
                          collate_fn=dl.collate_sparse_planes,
                          worker_init_fn=ev.worker_init,
                          pin_memory=device.type == "cuda")

    d = score(model, loader(args.round_offset, args.rounds), device)
    X, y, st, img = d["X"], d["label"], d["subtype"], d["image"]
    names = ev.SUBTYPES
    print(f"{len(val)} held-out images, rounds {args.round_offset}-"
          f"{args.round_offset + args.rounds - 1}: {len(y)} candidates",
          flush=True)

    # cross-fitting by image, for choosing between checkpoints
    half = np.random.default_rng(args.seed).permutation(len(val)) \
        < len(val) // 2
    fold = half[img]
    cf_refit, cf_bias = np.zeros(len(y)), np.zeros(len(y), bool)
    for f in (True, False):
        tr, te = fold != f, fold == f
        cf_refit[te] = fit(X[tr], y[tr], args.C).decision_function(X[te])
        cf_bias[te] = d["margin"][te] >= best_threshold(d["margin"][tr],
                                                         y[tr])
    report = {
        "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), _HERE),
        "population": {"images": len(val), "rounds": [
            args.round_offset, args.round_offset + args.rounds - 1],
            "candidates": int(len(y))},
        "uncalibrated": {"roc_auc": float(roc_auc_score(y, d["margin"])),
                         **summary(d["margin"], y, st, names)},
        "crossfit_bias_only": summary(cf_bias.astype(float), y, st,
                                      names, 0.5),
        "crossfit_refit": {"roc_auc": float(roc_auc_score(y, cf_refit)),
                           **summary(cf_refit, y, st, names)},
    }

    lr = fit(X, y, args.C)
    a, w, b = float(lr.coef_[0, 0]), lr.coef_[0, 1:], float(lr.intercept_[0])
    if a <= 0:
        sys.exit(f"glob coefficient {a:.4f} is not positive; not folding")
    report["fit"] = {"C": args.C, "intercept": b,
                     "coef": dict(zip(FEATURES, map(float, lr.coef_[0]))),
                     "in_sample_roc_auc": float(roc_auc_score(
                         y, lr.decision_function(X)))}

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    last = max(int(k.split(".")[1]) for k in sd if k.startswith("fc."))
    kw, kb = f"fc.{last}.weight", f"fc.{last}.bias"
    sd[kw] = sd[kw] * a
    sd[kb] = sd[kb] * a
    sd[kb][1] += b / 2
    sd[kb][0] -= b / 2
    sd["box_mix.weight"] = torch.tensor([list(map(float, w))],
                                        dtype=sd["box_mix.weight"].dtype)
    sd["box_mix.bias"] = torch.zeros_like(sd["box_mix.bias"])
    tmp = args.out + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, args.out)
    print(f"saved {args.out}", flush=True)

    # fold check: the folded model's margin is the fit's decision function
    folded = CorrNet.from_checkpoint(args.out, device).eval()
    n = min(2, args.rounds) * len(base)
    m = score(folded, loader(args.round_offset, min(2, args.rounds)),
              device)["margin"]
    ref = lr.decision_function(X[:n])
    diff = float(np.abs(m - ref).max())
    report["fold_check"] = {"candidates": n, "max_abs_diff": diff,
                            "same_decisions": int(np.sum(
                                (m >= 0) == (ref >= 0))),
                            "folded_accuracy": float(np.mean(
                                (m >= 0) == y[:n]))}
    report["out"] = os.path.relpath(os.path.abspath(args.out), _HERE)
    if diff > 1e-3:
        print(f"FOLD CHECK FAILED: max |margin - decision| = {diff:.3g}",
              flush=True)

    for k in ("uncalibrated", "crossfit_bias_only", "crossfit_refit"):
        r = report[k]
        print(f"{k:>19}: acc {r['accuracy']:.4f}"
              + (f"  AUC {r['roc_auc']:.4f}" if "roc_auc" in r else "")
              + f"  good {r['good_recall']:.3f}  " + "  ".join(
                  f"{n} {r[n]:.3f}" for n in WEIGHTS)
              + f"  (Section 5 weighted {r['section5_accuracy']:.4f})",
              flush=True)
    print("fit: " + "  ".join(f"{k} {v:+.4f}" for k, v in
                              report["fit"]["coef"].items())
          + f"  intercept {b:+.4f}", flush=True)
    print(f"fold check: max |margin - decision| {diff:.2e} over {n} "
          f"candidates, {report['fold_check']['same_decisions']}/{n} "
          f"identical decisions", flush=True)
    if args.report:
        with open(args.report + ".tmp", "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        os.replace(args.report + ".tmp", args.report)
        print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
