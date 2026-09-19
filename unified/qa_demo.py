"""Score the five demo samples with a unified-pipeline checkpoint.

Scores the five samples in samples/ (targets Good, Bad, Good, Good, Bad),
rendered with the qa_data transform the checkpoint was trained on, and
re-scores each with every box expanded by 35% -- an out-of-band
perturbation: each edge moves 17.5% of the box dimension, 3.5x the 0.05
band, so under the unified metric every expanded label is Bad. The
column shows how the model responds to it (a metric-faithful model flips
every Good sample to Bad); the band verdict of the expanded boxes, taken
against the bands around the sample's own boxes, is printed beside the
score so the record is self-explaining.

    QA_DATASET=coco python3 qa_demo.py artifacts_coco/best_ema_calibrated.pth
"""

import argparse
import json
import os
import sys

import cv2
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dataset as ds_conf
from qa_data import (MAPPING, box_rois, build_regions, get_sample,
                     is_label_negative)
from qa_model import load_any

TARGETS = {1: True, 2: False, 3: True, 4: True, 5: False}


def expand(payload, frac=0.35):
    out = {}
    for name, boxes in payload.items():
        out[name] = [[x1 - frac * (x2 - x1) / 2, y1 - frac * (y2 - y1) / 2,
                      x2 + frac * (x2 - x1) / 2, y2 + frac * (y2 - y1) / 2]
                     for x1, y1, x2, y2 in boxes]
    return out


def band_bad(reference, candidate, img_w, img_h):
    """Band-metric verdict of `candidate` against the 0.05 bands drawn
    around `reference` (qa_data's is_label_negative, the training-time
    metric); both are {class_name: [[x1, y1, x2, y2], ...]} in image
    pixels."""
    def indexed(payload):
        return {MAPPING[n]: [list(b) for b in bs]
                for n, bs in payload.items()}
    return is_label_negative(indexed(candidate),
                             build_regions(indexed(reference), img_w, img_h))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--samples", default=os.path.join(_HERE, "samples"))
    args = ap.parse_args()

    model = load_any(args.checkpoint, torch.device("cpu"))
    model.eval()

    def score(image, ann):
        img, planes = get_sample(image, ann)
        roi = None
        if model.per_box:
            rows = box_rois(ann, image.shape[1], image.shape[0])
            roi = torch.cat([torch.zeros(rows.shape[0], 1),
                             torch.from_numpy(rows)], 1)
        with torch.no_grad():
            logits = model(torch.from_numpy(planes).unsqueeze(0).float(),
                           torch.from_numpy(img).unsqueeze(0).float(),
                           boxes=roi)
            return float(torch.softmax(logits, 1)[0, 1])

    correct, exp_metric_bad, exp_scored_bad = 0, 0, 0
    for n in range(1, 6):
        with open(os.path.join(args.samples, f"sample_{n}.json")) as f:
            payload = json.load(f)
        image = cv2.imread(os.path.join(args.samples, f"sample_{n}.png"))
        if image is None:
            sys.exit(f"missing sample_{n}.png — run samples/fetch_samples.py")
        p = score(image, payload)
        expanded = expand(payload)
        p_exp = score(image, expanded)
        exp_bad = band_bad(payload, expanded, image.shape[1], image.shape[0])
        good = p >= 0.5
        ok = good == TARGETS[n]
        correct += ok
        exp_metric_bad += exp_bad
        exp_scored_bad += p_exp < 0.5
        target = "Good" if TARGETS[n] else "Bad "
        verdict = "Good" if good else "Bad "
        print(f"sample_{n}: P(good)={p:.3f} -> {verdict} (target {target}) "
              f"{'OK ' if ok else 'MISS'} | +35% exp: {p_exp:.3f} -> "
              f"{'Good' if p_exp >= 0.5 else 'Bad '} "
              f"(metric {'Bad' if exp_bad else 'Good'})")
    print(f"verdicts correct: {correct}/5; +35% expanded labels: metric Bad "
          f"{exp_metric_bad}/5, scored Bad {exp_scored_bad}/5")


if __name__ == "__main__":
    main()
