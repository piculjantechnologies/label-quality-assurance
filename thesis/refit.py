"""Held-out refit of a raster-head model's verdict layer, folded into the
checkpoint (README item 6; the last step of run.sh).

The verdict's good-minus-bad margin is glob + box_mix([smin, mean,
log1p(count)]) + bias: glob is the fully connected layers' margin, and the
three box_mix inputs are the raster head's smooth minimum, mean and count
over the claimed cells. Training fits these weights on the training
images, which the model has partly memorised; on unseen images the margin
shifts and the decision point moves with it. This
script refits glob's weight, the three box_mix weights and the intercept
by logistic regression on held-out data -- the training run's validation
images (its --seed and --val_split: 10% of the small set, 582 images),
with candidates drawn as the test protocol draws them (evaluate.py: the
thesis generator, detector share 0.1, good candidates inside the bands
without the train-time exact-ground-truth share), --rounds candidates per
image and label from draw round --round_offset on (by default rounds 4-7,
4656 candidates; train.py's fixed validation set uses rounds 0-3, so
the two never share a candidate) -- and folds the fit
in: the final fc layer is scaled by glob's coefficient, box_mix takes the
three box weights with bias 0, and the intercept goes into the final bias
as +b/2 on the good row and -b/2 on the bad row. P(good) >= 0.5 is then
the fit's zero point. The network and its inputs are unchanged; only
these five numbers move, and ROC-AUC can move with them because the
weighting of glob against the head changes.

The report also cross-fits: the images are split into two halves and each
half is scored by a fit on the other, so the pooled accuracy and AUC are
measured on images the fit did not see; they sit next to the
uncalibrated argmax and a cross-fitted threshold-only fit on the same
population. The large set (the VOC train split) is never touched.

    python3 refit.py artifacts/best_model.pth --seed 0 \\
        --out artifacts/best_model_refit.pth --report artifacts/refit.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data_loader import Data, fixed_rows
from neural_network import load_model
from train import pool_split, worker_init

_HERE = os.path.dirname(os.path.abspath(__file__))

FEATURES = ("glob", "smin", "mean", "log1p_count")


def score(model, loader, device, folded=None):
    """Per candidate: the model's own margin, the verdict inputs (glob and
    what box_mix reads) and the label; with folded, also that model's
    margin on the same candidates."""
    seen = {}
    hooks = [model.fc.register_forward_hook(
                 lambda m, i, o: seen.__setitem__("fc", o)),
             model.box_mix.register_forward_pre_hook(
                 lambda m, i: seen.__setitem__("box", i[0]))]
    rec = {k: [] for k in ("margin", "X", "label", "folded")}
    with torch.no_grad():
        for img, planes, y in loader:
            img = img.float().to(device)
            planes = planes.float().to(device)
            seen.clear()
            out = model(img, planes).float()
            g, box = seen["fc"].float(), seen["box"].float()
            rec["margin"] += (out[:, 1] - out[:, 0]).tolist()
            rec["X"] += torch.cat([(g[:, 1] - g[:, 0])[:, None], box],
                                  1).tolist()
            rec["label"] += y.tolist()
            if folded is not None:
                f = folded(img, planes).float()
                rec["folded"] += (f[:, 1] - f[:, 0]).tolist()
    for h in hooks:
        h.remove()
    return {k: np.array(v) for k, v in rec.items()}


def summary(s, y, thr=0.0):
    pg = s >= thr
    return {"accuracy": float(np.mean(pg == y)),
            "good_recall": float(pg[y == 1].mean()),
            "bad_recall": float((~pg[y == 0]).mean())}


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
    ap.add_argument("--pool",
                    default="~/fiftyone/voc-2012/validation/processed_annotations",
                    help="the training run's small-set pool")
    ap.add_argument("--seed", type=int, default=0,
                    help="the training run's --seed")
    ap.add_argument("--val_split", type=float, default=0.1,
                    help="the training run's --val_split")
    ap.add_argument("--rounds", type=int, default=4,
                    help="candidates per validation image and label")
    ap.add_argument("--round_offset", type=int, default=4,
                    help="first draw round (validation uses 0-3)")
    ap.add_argument("--detector_share", type=float, default=0.1)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(os.path.expanduser(args.checkpoint), device)
    if not model.raster_head:
        sys.exit("refit.py needs a raster_head checkpoint")
    # the training run's split: same seed, listing, filter and shuffle
    val, _ = pool_split(args.pool, args.seed, args.val_split)
    rows = fixed_rows(val, args.rounds, args.seed, offset=args.round_offset)
    image = np.array([k for _ in range(args.rounds)
                      for k in range(len(val)) for _ in (0, 1)])

    def loader(rs):
        return torch.utils.data.DataLoader(
            Data(rs, augment=False, detector_share=args.detector_share),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, worker_init_fn=worker_init)

    d = score(model, loader(rows), device)
    X, y = d["X"], d["label"]
    print(f"{len(val)} validation images, {args.rounds} rounds: {len(y)} "
          f"candidates ({int(y.sum())} good)", flush=True)

    # cross-fitting by image
    half = np.random.default_rng(args.seed).permutation(len(val)) \
        < len(val) // 2
    fold = half[image]
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
                         **summary(d["margin"], y)},
        "crossfit_bias_only": summary(cf_bias.astype(float), y, 0.5),
        "crossfit_refit": {"roc_auc": float(roc_auc_score(y, cf_refit)),
                           **summary(cf_refit, y)},
    }

    lr = fit(X, y, args.C)
    a, w, b = float(lr.coef_[0, 0]), lr.coef_[0, 1:], float(lr.intercept_[0])
    if a <= 0:
        sys.exit(f"glob coefficient {a:.4f} is not positive; not folding")
    report["fit"] = {"C": args.C, "intercept": b,
                     "coef": dict(zip(FEATURES, map(float, lr.coef_[0]))),
                     "in_sample_roc_auc": float(roc_auc_score(
                         y, lr.decision_function(X)))}

    sd = torch.load(os.path.expanduser(args.checkpoint), map_location="cpu",
                    weights_only=True)
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

    # fold check: on the first n refit candidates (redrawn from their row
    # seeds by a fresh loader), the folded model's margin equals the fit's
    # decision function of the original model's verdict inputs
    folded = load_model(args.out, device)
    n = min(len(rows), 4 * args.batch_size)
    c = score(model, loader(rows[:n]), device, folded=folded)
    ref = lr.decision_function(c["X"])
    diff = float(np.abs(c["folded"] - ref).max())
    report["fold_check"] = {"candidates": n, "max_abs_diff": diff,
                            "same_decisions": int(np.sum(
                                (c["folded"] >= 0) == (ref >= 0)))}
    report["out"] = os.path.relpath(os.path.abspath(args.out), _HERE)
    if diff > 1e-3:
        print(f"FOLD CHECK FAILED: max |margin - decision| = {diff:.3g}",
              flush=True)

    for k in ("uncalibrated", "crossfit_bias_only", "crossfit_refit"):
        r = report[k]
        print(f"{k:>19}: acc {r['accuracy']:.4f}"
              + (f"  AUC {r['roc_auc']:.4f}" if "roc_auc" in r else "")
              + f"  good {r['good_recall']:.3f}  bad {r['bad_recall']:.3f}",
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
