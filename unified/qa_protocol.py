"""Protocol evaluation over a held-out pool: the band metric, the five
error subtypes and their 31 combinations.

Independent of the training pipeline's candidate mixer: candidates here
are drawn by this file alone, labelled by the band metric alone, and
scored with a released checkpoint. Point the dataset's environment
variable at the HELD-OUT split (the pool the training run never sees):

    QA_DATASET=voc VOC_POOL=/path/to/voc-2012/train \\
        python3 qa_protocol.py artifacts_voc/best_ema_calibrated.pth \\
        --out artifacts_voc/protocol.json

Metric: each ground-truth box defines a two-sided band per coordinate --
outward by alpha and inward by beta times the box dimension. A candidate
is good when every class holds exactly as many boxes as regions and a
perfect region-to-box matching exists with every matched box in-band
(Kuhn's algorithm). Every candidate, drawn or corrupted, is labelled by
this metric: a corruption that lands inside the bands counts as good.

Generators (band-relative):
  good  every corner uniform inside its band; the exact ground truth
        verbatim with probability --p_exact
  A1    1..all boxes removed
  A2    one box, 1-4 coordinates pushed outside the band
  A3    1..all boxes given a wrong class
  B1    1-3 random boxes of classes present in the image
  B2    as B1 but classes absent from the image
Pool table: 50/50 good vs one uniformly chosen single error type.

This file carries its own copy of the band metric, evaluated on float
boxes in original-pixel space (bands clipped to [0, W] x [0, H]; a good
draw thinner than MIN_BOX_PX falls back to the ground-truth box). The training pipeline's qa_data judges
the same metric on the integer letterbox boxes the model is rendered
from, so the two can disagree for boxes within a pixel of a band edge or
of the image border; the protocol numbers are those of this file's
definition.
Combination table: the 31 non-empty subsets, --runs x --per_run each,
salted by the combination's canonical index so a subset run draws
identically to a full one.
"""

import argparse
import json
import os
import random
import statistics
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dataset as ds_conf
import qa_data
from qa_model import load_any

CANONICAL_COMBOS = [
    "A1", "A2", "A3", "A1A2", "A1A3", "A2A3", "A1A2A3",
    "B1", "B2", "B1B2",
    "A1B1", "A2B1", "A3B1", "A1A2B1", "A1A3B1", "A2A3B1", "A1A2A3B1",
    "A1B2", "A2B2", "A3B2", "A1A2B2", "A1A3B2", "A2A3B2", "A1A2A3B2",
    "A1B1B2", "A2B1B2", "A3B1B2", "A1A2B1B2", "A1A3B1B2", "A2A3B1B2",
    "A1A2A3B1B2",
]
SINGLETONS = ("A1", "A2", "A3", "B1", "B2")
MIN_BOX_PX = 4.0


# ------------------------------------------------------------- metric ------
def build_regions(annotation, img_w, img_h, alpha, beta):
    """Per-coordinate bands per ground-truth box, clipped to the image."""
    regions = {}
    for cls, boxes in annotation.items():
        for x, y, w, h in boxes:
            x1, y1, x2, y2 = x, y, x + w, y + h
            cw = lambda v: float(np.clip(v, 0.0, img_w))    # noqa: E731
            ch = lambda v: float(np.clip(v, 0.0, img_h))    # noqa: E731
            regions.setdefault(int(cls), []).append({
                "x1": (cw(x1 - w * alpha), cw(x1 + w * beta)),
                "y1": (ch(y1 - h * alpha), ch(y1 + h * beta)),
                "x2": (cw(x2 + w * alpha), cw(x2 - w * beta)),
                "y2": (ch(y2 + h * alpha), ch(y2 - h * beta)),
                "gt": (x1, y1, x2, y2),
                "wh": (max(1.0, float(w)), max(1.0, float(h))),
            })
    return regions


def _in_band(box_xyxy, region):
    for value, key in zip(box_xyxy, ("x1", "y1", "x2", "y2")):
        lo, hi = min(region[key]), max(region[key])
        if not lo <= value <= hi:
            return False
    return True


def _has_perfect_matching(adjacency, n_right):
    match = [-1] * n_right

    def augment(u, seen):
        for v in adjacency[u]:
            if not seen[v]:
                seen[v] = True
                if match[v] == -1 or augment(match[v], seen):
                    match[v] = u
                    return True
        return False

    return all(augment(u, [False] * n_right) for u in range(len(adjacency)))


def is_bad(candidate, regions, num_classes):
    """Bad unless every region holds exactly one in-band right-class box."""
    for cls in range(num_classes):
        boxes = candidate.get(cls, [])
        regs = regions.get(cls, [])
        if len(boxes) != len(regs):
            return True
        if not regs:
            continue
        xyxy = [(x, y, x + w, y + h) for x, y, w, h in boxes]
        adjacency = [[j for j, b in enumerate(xyxy) if _in_band(b, r)]
                     for r in regs]
        if any(not a for a in adjacency):
            return True
        if not _has_perfect_matching(adjacency, len(boxes)):
            return True
    return False


# --------------------------------------------------------- generators ------
def good_label(annotation, regions, rng, p_exact):
    """Corners drawn inside their bands; exact GT verbatim at p_exact."""
    if rng.random() < p_exact:
        return {c: [list(b) for b in bs] for c, bs in annotation.items()}
    out = {}
    for cls, regs in regions.items():
        for r in regs:
            draws = {}
            for key in ("x1", "y1", "x2", "y2"):
                lo, hi = min(r[key]), max(r[key])
                draws[key] = rng.uniform(lo, hi)
            x1, y1, x2, y2 = draws["x1"], draws["y1"], draws["x2"], draws["y2"]
            if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                x1, y1, x2, y2 = r["gt"]
            out.setdefault(cls, []).append([x1, y1, x2 - x1, y2 - y1])
    return out


def _random_box(rng, img_w, img_h):
    w = rng.uniform(0.05, 0.5) * img_w
    h = rng.uniform(0.05, 0.5) * img_h
    return [rng.uniform(0, img_w - w), rng.uniform(0, img_h - h), w, h]


def corrupt_combo(annotation, regions, rng, img_w, img_h, subtypes,
                  num_classes, alpha, p_exact):
    """Ordered error combination applied to a fresh good draw."""
    base = good_label(annotation, regions, rng, p_exact)
    reg_flat = [(c, r) for c in sorted(regions) for r in regions[c]]
    recs = []
    i = 0
    for c in sorted(base):
        for box in base[c]:
            recs.append([c, box, reg_flat[i][1] if i < len(reg_flat) else None])
            i += 1
    off_scale = max(alpha, 0.05)
    for st in subtypes:
        if st in ("A1", "A2", "A3") and not recs:
            continue
        if st == "A1":
            for i in sorted(rng.sample(range(len(recs)),
                                       rng.randint(1, len(recs))),
                            reverse=True):
                del recs[i]
        elif st == "A2":
            rec = recs[rng.randrange(len(recs))]
            if rec[2] is None:
                continue
            x, y, w, h = rec[1]
            coords = [x, y, x + w, y + h]
            r = rec[2]
            gw, gh = r["wh"]
            gdims = (gw, gh, gw, gh)
            dims = (img_w, img_h, img_w, img_h)
            keys = ("x1", "y1", "x2", "y2")
            for j in rng.sample(range(4), rng.randint(1, 4)):
                off = rng.uniform(0.1, 1.0) * off_scale * gdims[j]
                lo, hi = min(r[keys[j]]), max(r[keys[j]])
                outward_v = lo - off if j < 2 else hi + off
                inward_v = hi + off if j < 2 else lo - off
                ok_out = (outward_v >= 0.0) if j < 2 \
                    else (outward_v <= dims[j])
                pick = outward_v if (ok_out and rng.random() < 0.5) \
                    else inward_v
                coords[j] = min(max(pick, 0.0), float(dims[j]))
            x1, y1, x2, y2 = coords
            if x2 <= x1:
                x2 = min(x1 + MIN_BOX_PX, float(img_w))
                x1 = max(0.0, x2 - MIN_BOX_PX)
            if y2 <= y1:
                y2 = min(y1 + MIN_BOX_PX, float(img_h))
                y1 = max(0.0, y2 - MIN_BOX_PX)
            rec[1] = [x1, y1, x2 - x1, y2 - y1]
        elif st == "A3":
            for i in rng.sample(range(len(recs)), rng.randint(1, len(recs))):
                c = recs[i][0]
                recs[i][0] = rng.choice(
                    [k for k in range(num_classes) if k != c])
        else:  # B1 / B2
            present = sorted(annotation)
            absent = [k for k in range(num_classes) if k not in annotation]
            for _ in range(rng.randint(1, 3)):
                if st == "B1" or not absent:
                    cls = rng.choice(present)
                else:
                    cls = rng.choice(absent)
                recs.append([cls, _random_box(rng, img_w, img_h), None])
    combo = {}
    for c, box, _ in recs:
        combo.setdefault(c, []).append(box)
    return combo


# ------------------------------------------------------------ dataset ------
def pack(image, candidate, res):
    """Render one candidate exactly as training does: normalised RGB plus
    the sparse label planes, and the per-box RoI rows in letterbox
    pixels for a per-box checkpoint."""
    old_h, old_w = image.shape[:2]
    resized = qa_data.resize_keep_aspect(image, res)
    new_h, new_w = resized.shape[:2]
    oy, ox = (res - new_h) // 2, (res - new_w) // 2
    bbs, rows = {}, []
    for cls, boxes in candidate.items():
        for x, y, w, h in boxes:
            x1, y1 = min(x, x + w), min(y, y + h)
            x2, y2 = max(x, x + w), max(y, y + h)
            px1 = int(np.clip(x1 / old_w * new_w, 0, new_w - 1))
            py1 = int(np.clip(y1 / old_h * new_h, 0, new_h - 1))
            px2 = int(np.clip(x2 / old_w * new_w, 0, new_w - 1))
            py2 = int(np.clip(y2 / old_h * new_h, 0, new_h - 1))
            bbs.setdefault(int(cls), []).append([px1, py1, px2, py2])
            rows.append([px1 + ox, py1 + oy, px2 + ox, py2 + oy, int(cls)])
    planes = qa_data.render_planes(bbs, new_h, new_w, res)
    img = qa_data.normalize_image(resized.astype(np.float32), res)
    return (torch.from_numpy(np.ascontiguousarray(img).astype(np.float32)),
            torch.from_numpy(planes.astype(np.float32)),
            torch.from_numpy(np.array(rows, np.float32).reshape(-1, 5)))


def collate(batch):
    """Per-box rows are variable-length: concatenate them with the sample
    index prepended, the (N, 6) layout the model's `boxes` input wants."""
    img = torch.stack([b[0] for b in batch])
    planes = torch.stack([b[1] for b in batch])
    boxes = torch.cat([
        torch.cat([torch.full((b[2].shape[0], 1), float(i)), b[2]], 1)
        for i, b in enumerate(batch)], 0)
    labels = torch.as_tensor([b[3] for b in batch])
    return img, planes, boxes, labels


class EvalSet(Dataset):
    """Deterministic candidates for one experiment; every candidate is
    labelled by the band metric. mode None = the pool table (uniform
    single error type); a subtype tuple = that combination."""

    def __init__(self, pool, indices, mode, seed, salt, args, res):
        self.pool, self.indices = pool, indices
        self.mode, self.seed, self.salt = mode, seed, salt
        self.alpha, self.beta, self.p_exact = args.alpha, args.beta, args.p_exact
        self.res = res

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        entry = self.pool[self.indices[i]]
        path, annotation = entry[0], entry[1]
        rng = random.Random(self.seed * 1_000_003
                            + self.salt * 7_776_146_593 + i)
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape[:2]
        regions = build_regions(annotation, w, h, self.alpha, self.beta)
        if rng.random() < 0.5:
            candidate = good_label(annotation, regions, rng, self.p_exact)
        else:
            subtypes = self.mode or (rng.choice(SINGLETONS),)
            candidate = corrupt_combo(annotation, regions, rng, w, h,
                                      subtypes, ds_conf.NUM_CLASSES,
                                      self.alpha, self.p_exact)
        label = 0 if is_bad(candidate, regions, ds_conf.NUM_CLASSES) else 1
        return (*pack(image, candidate, self.res), label)


def run_pass(name, dataset, model, args, device):
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)
    labels, scores = [], []
    for img, planes, boxes, y in loader:
        with torch.autocast(device.type, enabled=args.amp
                            and device.type == "cuda"):
            logits = model(planes.to(device), img.to(device),
                           boxes=boxes.to(device) if model.per_box else None)
        scores += torch.softmax(logits.float(), 1)[:, 1].tolist()
        labels += y.tolist()
    acc = accuracy_score(labels, [s >= args.threshold for s in scores])
    print(f"[{name}] n={len(labels)}  good={sum(labels)}  acc {acc:.4f}",
          flush=True)
    return labels, scores


def parse_combo(text):
    subtypes = tuple(text[i:i + 2] for i in range(0, len(text), 2))
    for st in subtypes:
        if st not in SINGLETONS:
            raise ValueError(f"bad error combination {text!r}")
    return subtypes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--alpha", type=float, default=qa_data.ALPHA,
                    help="outward band, x box dimension")
    ap.add_argument("--beta", type=float, default=qa_data.BETA,
                    help="inward band, x box dimension")
    ap.add_argument("--p_exact", type=float, default=0.25,
                    help="exact-ground-truth share of good candidates")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=12000,
                    help="pool-table candidates (0 = the whole pool)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--per_run", type=int, default=800)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = ds_conf.load_pool()
    res = ds_conf.RES
    print(f"pool {len(pool)} images, {ds_conf.NUM_CLASSES} classes, "
          f"{res}px", flush=True)
    print(f"metric: alpha={args.alpha} beta={args.beta} "
          f"p_exact={args.p_exact} threshold={args.threshold}", flush=True)
    model = load_any(args.checkpoint, device)
    model.eval()

    results = {"config": vars(args) | {"pool": len(pool)},
               "pool_table": {}, "combinations": {}}

    indices = list(range(len(pool)))
    if args.limit and args.limit < len(indices):
        indices = sorted(random.Random(args.seed).sample(indices, args.limit))
    labels, scores = run_pass("pool", EvalSet(pool, indices, None, args.seed,
                                              0, args, res),
                              model, args, device)
    preds = [int(s >= args.threshold) for s in scores]
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0)
    results["pool_table"] = {
        "n": len(labels), "acc": accuracy_score(labels, preds),
        "auc": roc_auc_score(labels, scores),
        "p0": prec[0], "r0": rec[0], "f0": f1[0], "n0": int(support[0]),
        "p1": prec[1], "r1": rec[1], "f1": f1[1], "n1": int(support[1])}

    for name in CANONICAL_COMBOS:
        ci = CANONICAL_COMBOS.index(name)
        subtypes = parse_combo(name)
        accs = []
        for run in range(args.runs):
            salt = 1 + ci * 100 + run
            idx = random.Random(args.seed * 999_983 + salt).sample(
                range(len(pool)), min(args.per_run, len(pool)))
            labels, scores = run_pass(f"{name} {run + 1}/{args.runs}",
                                      EvalSet(pool, idx, subtypes, args.seed,
                                              salt, args, res),
                                      model, args, device)
            accs.append(accuracy_score(
                labels, [s >= args.threshold for s in scores]))
        results["combinations"][name] = {
            "runs": accs, "mean": statistics.mean(accs),
            "std": statistics.stdev(accs) if len(accs) > 1 else 0.0}

    mean = statistics.mean(v["mean"] for v in results["combinations"].values())
    results["combinations_mean"] = mean
    print(f"\npool acc {results['pool_table']['acc']:.4f}  "
          f"pool AUC {results['pool_table']['auc']:.4f}  "
          f"combination mean {mean:.4f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
            f.write("\n")
        print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
