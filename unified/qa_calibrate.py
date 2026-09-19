"""Val-only decision calibration for the unified pipeline: fold the
accuracy-optimal margin into the classifier bias.

Reproduces qa_train.py's validation population at the release recipe's
defaults (--p_edge and --swap_share must be repeated; runs trained with
--val_split, --limit, --severity_mix or --compose are not reproduced: the unified
negative mixer at the training shares, p_exact 0.5, the dataset's own
seed-0 shuffle and validation split, val dataset seed+1, bank and
co-occurrence from the training images), prints the AUC for comparison
with the training log's best EMA line, finds the
accuracy-optimal margin threshold t on that validation set, and writes a
copy of the checkpoint with [+t/2, -t/2] folded into the final bias --
so P(good) >= 0.5 realises the calibrated decision with the AUC
unchanged up to half-precision rounding of the shifted logits.
Calibration uses validation data only; the paper-protocol
test pool is untouched.

    QA_DATASET=voc python3 qa_calibrate.py artifacts_voc/best_ema.pth \
        --out artifacts_voc/best_ema_calibrated.pth --p_edge 0.25 --swap_share 0.1
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--p_exact", type=float, default=0.5)
    ap.add_argument("--p_edge", type=float, default=0.0)
    ap.add_argument("--swap_share", type=float, default=0.0)
    ap.add_argument("--detector_share", type=float, default=0.1)
    ap.add_argument("--ih_share", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    qa_data.good_label = functools.partial(qa_data.good_label,
                                           p_exact=args.p_exact,
                                           p_edge=args.p_edge)
    pool = ds_conf.load_pool()
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    n_val = max(10, int(len(pool) * ds_conf.VAL_SPLIT))
    val_pool, train_pool = pool[:n_val], pool[n_val:]
    qa_data.build_context(train_pool)
    print(f"{len(val_pool)} validation images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_any(args.checkpoint, device)
    model.eval()
    paths = []
    for img, packed, _, _ in val_pool:
        paths += [(img, packed, 1), (img, packed, 0)]
    val_ds = qa_data.Data(paths, augment=False,
                          detector_share=args.detector_share,
                          ih_share=args.ih_share,
                          swap_share=args.swap_share,
                          deterministic=True, seed=args.seed + 1,
                          sparse_planes=True, per_box=model.per_box)
    val_ds.recipe = recipe_state()      # reaches every loader worker
    loader = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=args.workers,
                        prefetch_factor=(1 if args.workers > 0 else None),
                        collate_fn=qa_data.collate_sparse,
                        worker_init_fn=worker_init)

    def margins_labels(m):
        margins, labels = [], []
        with torch.no_grad():
            for batch in loader:
                img, planes = qa_data.to_dense_batch(batch[0], batch[1],
                                                     batch[2], device)
                y = batch[3]
                roi = batch[-1][:, :6].to(device) if m.per_box else None
                with torch.amp.autocast("cuda",
                                        enabled=device.type == "cuda"):
                    logits = m(planes, img, boxes=roi)
                lg = logits.float()
                margins += (lg[:, 1] - lg[:, 0]).tolist()
                labels += y.tolist()
        return np.array(margins), np.array(labels)

    margins, labels = margins_labels(model)
    auc = roc_auc_score(labels, 1 / (1 + np.exp(-margins)))
    acc0 = float(np.mean((margins >= 0) == labels))
    print(f"uncalibrated: AUC {auc:.6f}  acc@0.5 {acc0:.4f} "
          f"(compare against the training log's best val line)")

    order = np.sort(margins)
    cands = np.concatenate([[order[0] - 1],
                            (order[:-1] + order[1:]) / 2,
                            [order[-1] + 1]])
    accs = [(float(np.mean((margins >= t) == labels)), t) for t in cands]
    best_acc, t = max(accs)
    print(f"val-optimal margin threshold t = {t:.4f}  ->  acc {best_acc:.4f}")

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sd["head.4.bias"] = sd["head.4.bias"].clone()
    sd["head.4.bias"][0] += t / 2
    sd["head.4.bias"][1] -= t / 2
    torch.save(sd, args.out)
    print(f"saved {args.out}")

    cal = load_any(args.out, device)
    cal.eval()
    m2, l2 = margins_labels(cal)
    auc2 = roc_auc_score(l2, 1 / (1 + np.exp(-m2)))
    acc2 = float(np.mean((m2 >= 0) == l2))
    print(f"calibrated:  AUC {auc2:.6f}  acc@0.5 {acc2:.4f}")


if __name__ == "__main__":
    main()
