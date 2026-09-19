"""Score the five demo samples with a CorrNet checkpoint.

Targets: Good, Bad, Good, Good, Bad. The +35% column re-scores each
sample with every box expanded by 35% — each edge moves outward by
17.5% of the box dimension; with the drawn boxes clipped to the image, as
the raster draws them, every box stays inside its image-clipped 0.2
uncertainty region, so the verdict should not change.

    python3 demo_samples.py <checkpoint.pth>
"""

import json
import os
import sys

import cv2
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data_loader import get_sample
from interactive_demo import COCO_CLASSES, load_any

TARGETS = {1: True, 2: False, 3: True, 4: True, 5: False}
NAME2IDX = {n: i for i, n in enumerate(COCO_CLASSES)}


def to_annotation(payload):
    ann = {}
    for cls_name, boxes in payload.items():
        idx = NAME2IDX[cls_name]
        for x1, y1, x2, y2 in boxes:
            ann.setdefault(idx, []).append([x1, y1, x2 - x1, y2 - y1])
    return ann


def expand(ann, frac=0.35):
    return {c: [[x - frac * w / 2, y - frac * h / 2,
                 w * (1 + frac), h * (1 + frac)] for x, y, w, h in lst]
            for c, lst in ann.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint",
                    help="CorrNet checkpoint, e.g. artifacts/best_model_refit.pth")
    args = ap.parse_args()
    device = torch.device("cpu")
    model = load_any(args.checkpoint, device)
    model.eval()

    def score(image, ann):
        cats, background = get_sample(640, image, ann)
        with torch.no_grad():
            logits = model(torch.from_numpy(cats).unsqueeze(0).float(),
                           torch.from_numpy(background).unsqueeze(0).float())
            return float(torch.softmax(logits, 1)[0, 1])

    base = os.path.join(_HERE, "samples")
    correct = 0
    for n in range(1, 6):
        with open(os.path.join(base, f"sample_{n}.json")) as f:
            payload = json.load(f)
        image = cv2.imread(os.path.join(base, f"sample_{n}.png"))
        assert image is not None, (
            f"missing samples/sample_{n}.png -- run "
            f"python3 samples/fetch_samples.py first")
        ann = to_annotation(payload)
        p = score(image, ann)
        p_exp = score(image, expand(ann))
        good = p >= 0.5
        ok = good == TARGETS[n]
        correct += ok
        target = "Good" if TARGETS[n] else "Bad "
        verdict = "Good" if good else "Bad "
        print(f"sample_{n}: P(good)={p:.3f} -> {verdict} (target {target}) "
              f"{'OK ' if ok else 'MISS'} | +35% exp: {p_exp:.3f}")
    print(f"verdicts correct: {correct}/5")


if __name__ == "__main__":
    main()
