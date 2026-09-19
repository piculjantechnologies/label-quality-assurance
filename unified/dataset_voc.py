"""Dataset configuration: Pascal VOC 2012, 20 classes, 224-px inputs.

One of the two dataset configurations, selected by QA_DATASET=voc in
`dataset.py`; everything else in the pipeline (qa_data, qa_model,
qa_train, qa_calibrate) is dataset-agnostic.

Training pool: the VOC validation split (the thesis's carefully-annotated
*small set*, 5823 images); the train split (5717 images) is the held-out
large set used for evaluation. The thesis's 80/20 model-selection split
is kept via VAL_SPLIT = 0.2.
"""

import glob
import os

import numpy as np

RES = 224            # letterbox size
FUSION_RES = 28      # stride-8 fusion grid at 224
VAL_SPLIT = 0.2      # the thesis's 80/20 model-selection split

POOL_DIR = os.path.expanduser(
    os.environ.get("VOC_POOL", "~/fiftyone/voc-2012/validation"))

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor']
NUM_CLASSES = len(CLASSES)

# torchvision's detection models emit COCO category names; map the six
# whose VOC spelling differs (other COCO classes are dropped upstream by
# the MAPPING membership test).
DETECTOR_NAME_MAP = {'motorcycle': 'motorbike', 'airplane': 'aeroplane',
                     'couch': 'sofa', 'potted plant': 'pottedplant',
                     'dining table': 'diningtable', 'tv': 'tvmonitor'}


def class_names():
    return CLASSES


def load_pool():
    """[(image_path, packed {cls: [[x, y, w, h]]}, img_w, img_h), ...]

    The packed .npy annotations carry original-pixel x, y, w, h per
    class index; image dimensions come from the image headers.
    """
    from PIL import Image
    ann_dir = os.path.join(POOL_DIR, "processed_annotations")
    img_dir = os.path.join(POOL_DIR, "data")
    pool = []
    for f in sorted(glob.glob(os.path.join(ann_dir, "*.npy"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        img = os.path.join(img_dir, stem + ".jpg")
        if not os.path.isfile(img):
            continue
        packed_raw = np.load(f, allow_pickle=True).item()
        packed = {int(c): [list(map(float, b)) for b in boxes]
                  for c, boxes in packed_raw.items()
                  if int(c) < NUM_CLASSES and boxes}
        if not packed:
            continue
        with Image.open(img) as im:      # header only, no full decode
            w, h = im.size
        pool.append((img, packed, w, h))
    return pool
