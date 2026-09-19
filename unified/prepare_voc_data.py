"""Convert a fiftyone Pascal VOC 2012 export to per-image annotation files.

Input  (created by download_pascal_voc_2012_dataset.py):
    <dataset_dir>/<split>/data/<name>.jpg
    <dataset_dir>/<split>/labels.json   {"classes": [20 names],
                                         "labels": {name: [{"label": cls_idx,
                                            "bounding_box": [x,y,w,h] in 0..1}]}}
Output (what dataset_voc.load_pool consumes):
    <dataset_dir>/<split>/processed_annotations/<name>.npy
        dict {class_idx: [[x, y, w, h], ...]} in absolute pixels

Run:  python3 prepare_voc_data.py --split train
      python3 prepare_voc_data.py --split validation
"""

import os
import json
import argparse

import cv2
import numpy as np

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def main():
    ap = argparse.ArgumentParser(description="Prepare VOC-2012 annotations")
    ap.add_argument("--dataset_dir", default="~/fiftyone/voc-2012",
                    help="fiftyone export root (contains train/ and validation/)")
    ap.add_argument("--split", choices=["train", "validation"], required=True)
    args = ap.parse_args()

    root = os.path.join(os.path.expanduser(args.dataset_dir), args.split)
    with open(os.path.join(root, "labels.json")) as f:
        meta = json.load(f)
    assert meta["classes"] == VOC_CLASSES, "unexpected class list/order"

    out_dir = os.path.join(root, "processed_annotations")
    os.makedirs(out_dir, exist_ok=True)

    labels = meta["labels"]
    for k, (name, anns) in enumerate(sorted(labels.items())):
        img = cv2.imread(os.path.join(root, "data", name + ".jpg"))
        if img is None:
            print(f"skip {name}: image missing")
            continue
        h, w = img.shape[:2]
        packed = {}
        for a in anns:
            x, y, bw, bh = a["bounding_box"]
            packed.setdefault(int(a["label"]), []).append(
                [x * w, y * h, bw * w, bh * h])
        np.save(os.path.join(out_dir, name + ".npy"), packed)
        if (k + 1) % 500 == 0:
            print(f"{k + 1}/{len(labels)}", flush=True)
    print(f"done: {len(labels)} files -> {out_dir}")


if __name__ == "__main__":
    main()
