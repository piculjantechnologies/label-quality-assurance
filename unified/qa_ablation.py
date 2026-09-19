"""Input ablation for a unified-pipeline checkpoint: validation AUC with
the true, mismatched, and blank image.

Rebuilds qa_train.py's validation population at the release recipe's
defaults (same shuffle, split, mixer shares and seeds; --p_edge and
--swap_share must be repeated, and runs trained with --val_split,
--limit or --severity_mix are not reproduced), then scores each batch
three ways on
identical (planes, y): the true image, a mismatched image, and a zeroed
image (blank). A model that reads the image collapses on 'mismatched';
'blank' measures what the label branch alone recovers.

The population holds one good-requested and one bad-requested candidate
per image (labels are re-derived by the metric, so an in-band detector
candidate in the bad slot counts as good), adjacent and in that order,
so the mismatched variant rolls the batch by
TWO: rolling by one would hand every second row its own image back, and
those rows are exactly the negatives -- a half-no-op control that
corrupts only the positives and inverts the ranking by construction.

The sampler overrides and the training-pool context reach the loader
workers through qa_train's recipe_state / worker_init, whatever the
multiprocessing start method.

    QA_DATASET=voc python3 qa_ablation.py artifacts_voc/best_ema_calibrated.pth \
        --p_edge 0.25 --swap_share 0.1
"""

import argparse
import functools
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dataset as ds_conf
import qa_data
from qa_model import load_any
from qa_train import recipe_state, worker_init


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--p_exact", type=float, default=0.5)
    ap.add_argument("--p_edge", type=float, default=0.0)
    ap.add_argument("--swap_share", type=float, default=0.0)
    ap.add_argument("--detector_share", type=float, default=0.1)
    ap.add_argument("--ih_share", type=float, default=0.5)
    ap.add_argument("--compose", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    # the pair stride the mismatched roll relies on: at a batch of 2 the
    # roll is the identity, and an odd batch breaks the pairing itself
    if args.batch_size < 4 or args.batch_size % 2:
        sys.exit("--batch_size must be even and at least 4: the "
                 "positive/negative pairs are adjacent and the "
                 "mismatched control rolls by two")

    qa_data.good_label = functools.partial(qa_data.good_label,
                                           p_exact=args.p_exact,
                                           p_edge=args.p_edge)
    pool = ds_conf.load_pool()
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    n_val = max(10, int(len(pool) * ds_conf.VAL_SPLIT))
    val_pool, train_pool = pool[:n_val], pool[n_val:]
    qa_data.build_context(train_pool)

    paths = []
    for img, packed, _, _ in val_pool:
        paths += [(img, packed, 1), (img, packed, 0)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_any(args.checkpoint, device)
    model.eval()

    ds = qa_data.Data(paths, augment=False,
                      detector_share=args.detector_share,
                      ih_share=args.ih_share, compose=args.compose,
                      swap_share=args.swap_share,
                      deterministic=True, seed=args.seed + 1,
                      sparse_planes=True, per_box=model.per_box)
    ds.recipe = recipe_state()
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.workers,
                        prefetch_factor=(1 if args.workers > 0 else None),
                        collate_fn=qa_data.collate_sparse,
                        worker_init_fn=worker_init)

    scores = {"true": [], "mismatched": [], "blank": []}
    labels = {"true": [], "mismatched": [], "blank": []}
    with torch.no_grad():
        for batch in loader:
            img, planes = qa_data.to_dense_batch(batch[0], batch[1],
                                                 batch[2], device)
            y = batch[3]
            roi = batch[-1][:, :6].to(device) if model.per_box else None
            variants = {"true": img, "blank": torch.zeros_like(img)}
            if img.shape[0] >= 4:
                # a trailing batch of two rows would make the roll-by-two
                # the identity, so such a tail is scored only true/blank
                variants["mismatched"] = torch.roll(img, 2, dims=0)
            for name, im in variants.items():
                with torch.amp.autocast("cuda",
                                        enabled=device.type == "cuda"):
                    logits = model(planes, im, boxes=roi)
                scores[name] += torch.softmax(logits.float(),
                                              1)[:, 1].tolist()
                labels[name] += y.tolist()

    print(f"{args.checkpoint}  ({len(labels['true'])} val samples)")
    for name in ("true", "mismatched", "blank"):
        sc = np.array(scores[name])
        lab = np.array(labels[name])
        auc = roc_auc_score(lab, sc)
        acc = float(np.mean((sc >= 0.5) == lab))
        print(f"{name:>12}: AUC {auc:.3f}  acc {acc:.3f}  "
              f"mean P(good) pos {sc[lab == 1].mean():.3f} "
              f"neg {sc[lab == 0].mean():.3f}")


if __name__ == "__main__":
    main()
