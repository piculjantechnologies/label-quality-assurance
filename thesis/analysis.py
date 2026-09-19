"""Measurement suite for the release (README item 3 — reporting only).

Two reports, both on the large set (the VOC train split); the ablation
uses the evaluation draw unchanged, the per-op report forces one Table 8.2
op per candidate:

  --ablation    score identical candidates -- one bad and one good request
                for each of --n_images images (500: 1000 candidates) --
                with the true image, the image rolled by two within the
                batch (mismatched), and a zeroed image (blank). A model
                whose verdicts come from the label branch alone produces
                bit-identical probabilities in all three conditions; an
                image-grounded model degrades.
  --per_op      bad-recall per thesis error type (Table 8.2): single-op
                corruptions, one op forced per candidate, --per_op_n (150)
                candidates per op; erase also reports its empty-label and
                boxes-remain cases.

The model is built before seeding, so the draw does not depend on it.

    python3 analysis.py --ablation --per_op \
        --ckpt artifacts/best_model_refit.pth --out artifacts/analysis_release.json
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from data_loader import (Data, ann_to_img, build_regions, corrupt,
                         normalize_image, render_planes, resize_keep_aspect)
from neural_network import load_model
from train import set_seed, worker_init

import cv2

OPS = ('erase', 'swap', 'translate', 'resize', 'crop', 'combine', 'split',
       'jitter')


def pool_files(pool, seed, limit):
    pool_dir = os.path.expanduser(pool)
    files = sorted(os.path.join(pool_dir, f) for f in os.listdir(pool_dir)
                   if f.endswith(".npy"))
    files = [f for f in files
             if any(len(v) for v in
                    np.load(f, allow_pickle=True).item().values())]
    random.Random(seed).shuffle(files)
    return files[:limit] if limit else files


def load_image_and_regions(path):
    image = cv2.imread(ann_to_img(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    old_h, old_w = image.shape[:2]
    image = resize_keep_aspect(image)
    new_h, new_w = image.shape[:2]
    packed = np.load(path, allow_pickle=True).item()
    boxes = {key: [] for key in range(20)}
    for key, segs in packed.items():
        if int(key) >= 20:
            continue
        for x, y, w, h in segs:
            boxes[int(key)].append(
                [np.clip(x / old_w * new_w, 0, new_w - 1),
                 np.clip(y / old_h * new_h, 0, new_h - 1),
                 np.clip((x + w) / old_w * new_w, 0, new_w - 1),
                 np.clip((y + h) / old_h * new_h, 0, new_h - 1)])
    return image, build_regions(boxes, new_w, new_h), new_w, new_h


@torch.no_grad()
def score(model, device, image, bbs, new_h, new_w):
    img = torch.from_numpy(normalize_image(image)).float().unsqueeze(0)
    planes = torch.from_numpy(render_planes(bbs, new_h, new_w)) \
        .float().unsqueeze(0)
    logits = model(img.to(device), planes.to(device))
    return float(torch.softmax(logits, 1)[0, 1])


def run_ablation(model, device, files, batch_size, workers):
    rows = []
    for f in files:
        rows += [(f, 0), (f, 1)]
    ds = Data(rows, augment=False)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers,
        worker_init_fn=worker_init)
    conds = {"true": ([], []), "mismatched": ([], []), "blank": ([], [])}
    max_diff = 0.0
    with torch.no_grad():
        for img, planes, y in loader:
            img = img.float().to(device)
            planes = planes.float().to(device)
            per_cond = {}
            # Roll by 2, not 1: rows alternate (bad, good) requests of the
            # same file, so a shift of 1 would hand half the candidates
            # their own image back.
            variants = [("true", img), ("blank", torch.zeros_like(img))]
            if img.shape[0] >= 4:
                # a trailing batch of two rows would make the roll-by-two
                # the identity, so such a tail is scored only true/blank
                variants.insert(1, ("mismatched", torch.roll(img, 2, 0)))
            for name, im in variants:
                p = torch.softmax(model(im, planes), 1)[:, 1]
                conds[name][0].extend(p.cpu().tolist())
                conds[name][1].extend(y.tolist())
                per_cond[name] = p
            max_diff = max([max_diff] + [
                float((per_cond["true"] - per_cond[k]).abs().max())
                for k in ("mismatched", "blank") if k in per_cond])
    from sklearn.metrics import roc_auc_score
    out = {}
    for name, (probs, labels) in conds.items():
        preds = [int(p >= 0.5) for p in probs]
        out[name] = {"n": len(labels),
                     "auc": roc_auc_score(labels, probs),
                     "acc": float(np.mean([p == l
                                           for p, l in zip(preds, labels)]))}
    out["max_abs_prob_diff_vs_true"] = max_diff
    return out


def run_per_op(model, device, files, per_op):
    out = {}
    for op in OPS:
        probs, empty = [], []
        for path in files:
            if len(probs) >= per_op:
                break
            image, regions, new_w, new_h = load_image_and_regions(path)
            bbs, ops = corrupt(regions, new_w, new_h, return_ops=True,
                               force_ops=[op])
            # `not ops` guards a generator that could not apply the op to
            # this image; an empty label set is a legitimate candidate
            # (erase removes every box) and must be scored, not skipped
            if not ops:
                continue
            probs.append(score(model, device, image, bbs, new_h, new_w))
            empty.append(not any(len(v) for v in bbs.values()))
        rec = lambda ps: (float(np.mean([p < 0.5 for p in ps]))  # noqa: E731
                          if ps else None)
        out[op] = {"n": len(probs),
                   "bad_recall": rec(probs),
                   "mean_p_good": float(np.mean(probs)) if probs else None}
        if any(empty):
            # an op that can empty the label set (erase) mixes two very
            # different populations; report them separately as well
            out[op]["n_empty_label"] = int(sum(empty))
            out[op]["bad_recall_empty_label"] = rec(
                [p for p, e in zip(probs, empty) if e])
            out[op]["bad_recall_boxes_remain"] = rec(
                [p for p, e in zip(probs, empty) if not e])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/best_model_refit.pth")
    ap.add_argument("--pool",
                    default="~/fiftyone/voc-2012/train/processed_annotations")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--per_op", action="store_true")
    ap.add_argument("--n_images", type=int, default=500)
    ap.add_argument("--per_op_n", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.batch_size < 4 or args.batch_size % 2:
        ap.error("--batch_size must be even and at least 4: rows are "
                 "(bad, good) pairs and the mismatched control rolls by two")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(os.path.expanduser(args.ckpt), device)
    set_seed(args.seed)     # after the model: the draw must not depend on it

    results = {"ckpt": args.ckpt, "seed": args.seed}
    if args.ablation:
        files = pool_files(args.pool, args.seed, args.n_images)
        results["ablation"] = run_ablation(model, device, files,
                                           args.batch_size, args.workers)
        print("ablation:", json.dumps(results["ablation"], indent=1))
    if args.per_op:
        set_seed(args.seed)
        files = pool_files(args.pool, args.seed + 2, 0)
        results["per_op"] = run_per_op(model, device, files, args.per_op_n)
        print("per_op:", json.dumps(results["per_op"], indent=1))

    if args.out:
        with open(args.out, "w") as f:
            results["args"] = vars(args)
            json.dump(results, f, indent=2)
            f.write("\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
