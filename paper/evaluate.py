"""Evaluate a checkpoint on the paper's test protocol (Section 5, Table 2).

The paper trains on COCO val2017 and tests on train2017. Every filtered
train2017 image contributes one candidate, good or bad with equal
probability; a bad candidate carries one error subtype drawn as in
Section 5 (50% type A with A1/A2/A3 uniform, 50% type B with B1/B2
uniform), and a good candidate is a draw inside the uncertainty regions.
This script builds that pool with the paper's generator in data_loader.py
(box-scaled regions, README interpretation 2; good candidates drawn from
the regions only, without the exact-ground-truth share of README item 3),
scores it with a checkpoint, and reports accuracy, per-class
precision / recall / F1 in the layout of Table 2, and ROC-AUC. A
candidate is predicted good when P(good) >= 0.5. Each bad candidate's
subtype is recorded, and the report adds its recall and a Table 4-style
accuracy: the mean of good recall and that subtype's recall, which is
what Table 4's single-subtype rows measure on their balanced samples.
Draws are seeded per (seed, index), so a rerun with the same --seed
scores the same pool whatever --batch_size and --workers are.
--scores saves every candidate's P(good), label and subtype.

    python3 evaluate.py --checkpoint artifacts/best_model_refit.pth \\
        --images /path/to/coco/train2017 \\
        --annotations /path/to/coco/annotations/instances_train2017.json \\
        --out artifacts/eval.json
"""

import argparse
import functools
import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader, Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import data_loader as dl
from coco_corrnet import CorrNet

_GOOD = dl.good           # the generator's positive draw, before any override
_CORRUPT = dl.corrupt
SUBTYPES = ("good", "A1", "A2", "A3", "B1", "B2")
_last_subtype = [None]


def _recording_corrupt(*args, **kwargs):
    """corrupt(), noting the subtype it drew for this process's last
    bad candidate; the draw itself is unchanged."""
    out, subtype = _CORRUPT(*args, **kwargs)
    _last_subtype[0] = subtype
    return out, subtype


dl.corrupt = _recording_corrupt   # LabelQualityDataset calls it by name


class TestPool(Dataset):
    """One candidate per image: good or bad by a seeded coin.

    Wraps LabelQualityDataset, whose even index 2i yields image i's good
    candidate and odd index 2i+1 its bad one, so each image's candidate is
    the one the training-time generator would draw for that class.
    """

    def __init__(self, samples, images_dir, size, seed, p_exact):
        self.base = dl.LabelQualityDataset(samples, images_dir, size,
                                           seed=seed, sparse_planes=True)
        coin = random.Random(seed)
        self.good = [coin.random() < 0.5 for _ in samples]
        self.p_exact = p_exact

    def __len__(self):
        return len(self.good)

    def __getitem__(self, i):
        item = self.base[2 * i + (0 if self.good[i] else 1)]
        subtype = "good" if self.good[i] else _last_subtype[0]
        return (*item, SUBTYPES.index(subtype))


def install_p_exact(p_exact):
    """Set the positive draw's exact-ground-truth share for this process
    (corrupt() draws its base label through the same function)."""
    dl.good = functools.partial(_GOOD, p_exact=p_exact)


def worker_init(worker_id):
    dl.cv2_single_thread_worker(worker_id)
    info = torch.utils.data.get_worker_info()
    if info is not None:
        install_p_exact(info.dataset.p_exact)


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",
                    default=os.path.join(_HERE, "artifacts", "best_model_refit.pth"))
    ap.add_argument("--images", required=True,
                    help="COCO train2017 image directory")
    ap.add_argument("--annotations", required=True,
                    help="COCO instances_train2017.json")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_exact", type=float, default=0.0,
                    help="share of good candidates that are the ground "
                         "truth itself (0 = the paper's region draw)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N images (0 = all)")
    ap.add_argument("--out", default="")
    ap.add_argument("--scores", default="",
                    help="also save per-candidate P(good), label and "
                         "subtype to this .npz")
    args = ap.parse_args()

    if not os.path.isdir(args.images):
        sys.exit(f"--images is not a directory: {args.images}")
    if not os.path.isfile(args.annotations):
        sys.exit(f"--annotations is not a file: {args.annotations}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorrNet.from_checkpoint(args.checkpoint, device)
    model.eval()

    samples, _ = dl.load_coco(args.annotations)
    if args.limit:
        samples = samples[:args.limit]
    pool = TestPool(samples, args.images, args.size, args.seed, args.p_exact)
    install_p_exact(args.p_exact)
    n_good = sum(pool.good)
    print(f"{len(pool)} images, one candidate each: {n_good} good, "
          f"{len(pool) - n_good} bad (seed {args.seed}, "
          f"p_exact {args.p_exact})", flush=True)

    loader = DataLoader(pool, batch_size=args.batch_size,
                        num_workers=args.workers,
                        collate_fn=dl.collate_sparse_planes,
                        worker_init_fn=worker_init,
                        pin_memory=device.type == "cuda",
                        prefetch_factor=4 if args.workers > 0 else None)

    scores, labels, subtypes = [], [], []
    t0 = time.time()
    n_batches = len(loader)
    with torch.no_grad():
        for b, batch in enumerate(loader, 1):
            cats, background, y, st = batch
            cats = dl.planes_to_dense(cats, background, device)
            logits = model(cats, background.to(device))
            scores += torch.softmax(logits.float(), 1)[:, 1].tolist()
            labels += y.tolist()
            subtypes += st.tolist()
            if b % 200 == 0 or b == n_batches:
                done = len(labels)
                rate = done / (time.time() - t0)
                eta = (len(pool) - done) / rate if rate else 0.0
                print(f"[{done}/{len(pool)}] {rate:.1f} candidates/s, "
                      f"eta {eta / 60:.1f} min", flush=True)

    scores = np.array(scores)
    labels = np.array(labels)
    subtypes = np.array(subtypes)
    preds = (scores >= 0.5).astype(int)
    good_recall = float(preds[labels == 1].mean())
    per_subtype = {}
    for k, name in enumerate(SUBTYPES[1:], 1):
        hit = preds[subtypes == k] == 0
        per_subtype[name] = {
            "support": int(hit.size),
            "recall": float(hit.mean()),
            "table4_accuracy": (good_recall + float(hit.mean())) / 2}
    report = classification_report(labels, preds, labels=[0, 1],
                                   target_names=["bad", "good"], digits=4,
                                   output_dict=True, zero_division=0)
    result = {
        "checkpoint": os.path.relpath(os.path.abspath(args.checkpoint), _HERE),
        "args": {k: v for k, v in vars(args).items()
                 if k not in ("images", "annotations", "checkpoint", "out",
                              "scores")},
        "images": int(len(labels)),
        "support": {"bad": int((labels == 0).sum()),
                    "good": int((labels == 1).sum())},
        "accuracy": float((preds == labels).mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "classes": {name: {k: float(report[name][k])
                           for k in ("precision", "recall", "f1-score")}
                    for name in ("bad", "good")},
        "macro_avg": {k: float(report["macro avg"][k])
                      for k in ("precision", "recall", "f1-score")},
        "weighted_avg": {k: float(report["weighted avg"][k])
                         for k in ("precision", "recall", "f1-score")},
        "subtypes": per_subtype,
    }
    print(classification_report(labels, preds, labels=[0, 1],
                                target_names=["bad", "good"], digits=4,
                                zero_division=0), flush=True)
    print("subtype  support  recall  Table 4-style accuracy")
    for name, r in per_subtype.items():
        print(f"{name:>7}  {r['support']:7d}  {r['recall']:.4f}  "
              f"{r['table4_accuracy']:.4f}")
    print(f"accuracy {result['accuracy']:.4f}  ROC-AUC "
          f"{result['roc_auc']:.4f}  ({len(labels)} candidates, "
          f"{(time.time() - t0) / 60:.1f} min)", flush=True)
    if args.scores:
        np.savez_compressed(args.scores, p_good=scores, label=labels,
                            subtype=subtypes,
                            subtype_names=np.array(SUBTYPES))
        print(f"wrote {args.scores}", flush=True)
    if args.out:
        atomic_json(args.out, result)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
