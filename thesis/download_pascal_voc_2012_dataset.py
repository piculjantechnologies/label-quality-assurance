"""Download Pascal VOC 2012 from the fiftyone dataset zoo (its train and
validation splits, exported under ~/fiftyone/voc-2012/); prepare_voc_data.py
then packs each split's annotations."""
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("voc-2012")
