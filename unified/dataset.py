"""Dataset selector for the unified pipeline.

Every other file imports `dataset` for what a dataset must supply
(CLASSES, NUM_CLASSES, RES, FUSION_RES, VAL_SPLIT, DETECTOR_NAME_MAP,
load_pool). Set QA_DATASET to pick the configuration:

    QA_DATASET=coco   COCO 2017, 80 classes, 640-px inputs
                      (also set COCO_IMAGES / COCO_ANNOTATIONS)
    QA_DATASET=voc    Pascal VOC 2012, 20 classes, 224-px inputs
                      (also set VOC_POOL)
"""

import os as _os

_name = _os.environ.get("QA_DATASET", "").strip().lower()
if _name == "coco":
    from dataset_coco import *          # noqa: F401,F403
elif _name == "voc":
    from dataset_voc import *           # noqa: F401,F403
else:
    raise SystemExit(
        "set QA_DATASET=coco or QA_DATASET=voc (see dataset.py)")
