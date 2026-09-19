"""Dataset configuration: COCO 2017, 80 classes, 640-px inputs.

One of the two dataset configurations, selected by QA_DATASET=coco in
`dataset.py`; everything else in the pipeline (qa_data, qa_model,
qa_train, qa_calibrate) is dataset-agnostic.

Training pool: the COCO val2017 split (the paper trains and validates on
val2017 and reserves train2017 as its test set); whole-image crowd
exclusion as in the paper; seed-0 shuffle, 5% held out for validation.
"""

import json
import os
from collections import defaultdict

import numpy as np

RES = 640            # letterbox size
FUSION_RES = 40      # stride-16 fusion grid at 640
VAL_SPLIT = 0.05     # 5% of the pool held out for model selection

IMAGES = os.path.expanduser(os.environ.get("COCO_IMAGES", ""))
ANNOTATIONS = os.path.expanduser(os.environ.get("COCO_ANNOTATIONS", ""))

# torchvision's detection models emit COCO category names directly; the
# only normalisation needed is identity.
DETECTOR_NAME_MAP = {}

_CLASSES = None
_SAMPLES = None


def _load():
    global _CLASSES, _SAMPLES
    if _SAMPLES is not None:
        return
    if not ANNOTATIONS or not os.path.isfile(ANNOTATIONS):
        raise SystemExit("set COCO_ANNOTATIONS to the instances json")
    if not IMAGES or not os.path.isdir(IMAGES):
        raise SystemExit("set COCO_IMAGES to the image directory")
    with open(ANNOTATIONS) as f:
        coco = json.load(f)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    _CLASSES = [c["name"] for c in cats]
    cat2idx = {c["id"]: i for i, c in enumerate(cats)}
    dims = {im["id"]: (im["file_name"], im["width"], im["height"])
            for im in coco["images"]}
    packed = defaultdict(dict)
    crowd = set()
    for a in coco["annotations"]:
        if a.get("iscrowd"):
            crowd.add(a["image_id"])
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        packed[a["image_id"]].setdefault(
            cat2idx[a["category_id"]], []).append([x, y, w, h])
    _SAMPLES = []
    for img_id, ann in sorted(packed.items()):
        if img_id in crowd or not ann:
            continue
        fn, w, h = dims[img_id]
        _SAMPLES.append((os.path.join(IMAGES, fn), dict(ann), w, h))


def class_names():
    _load()
    return _CLASSES


def load_pool():
    """[(image_path, packed {cls: [[x, y, w, h]]}, img_w, img_h), ...]"""
    _load()
    return list(_SAMPLES)


# The 80 COCO class names (categories in ascending id order), embedded so
# the demo tools work without the annotations file.
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]
NUM_CLASSES = len(CLASSES)
