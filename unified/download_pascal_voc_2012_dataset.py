"""Download Pascal VOC 2012 (train and validation splits) through the
fiftyone zoo into ~/fiftyone/voc-2012/; prepare_voc_data.py then packs the
annotations."""
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("voc-2012")
