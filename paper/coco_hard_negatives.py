"""Image-hard negatives: bad labels statistically ordinary in label space
(README item 6; the recipe mixes them 50/50 with the paper's generator via
corrupt_mixed and --paper_mix 0.5).

Without this module most negatives are detectable from label geometry
alone, so the training objective barely rewards checking the image. The
leak is concentrated in the subtypes whose output geometry differs from
real labels:

  B1/B2  _random_box draws uniform 5-50%-of-image boxes -- nothing like
         the class-conditional geometry of real COCO boxes.
  A1     deleting up to every box leaves implausibly sparse labels.

This module keeps the paper's error taxonomy but makes each negative's
geometry indistinguishable from a good label's:

  B1/B2  added boxes are real ground-truth boxes of the same class from
         the pool's images (normalized coords, rescaled here; the bank
         covers every image of the annotation file); B2's class
         is drawn context-plausibly via coco_a3_plausible.
  A1     deletes exactly one box.
  A3     unchanged from coco_a3_plausible (already label-plausible).
  A2     unchanged (a small violation is not detectable from the label
         alone).
  R      a subtype outside the paper's taxonomy: one box relocated to a
         random position (size kept, off any ground-truth region where one of
         20 placements finds room) -- a
         plausible box that covers no object, detectable only by looking
         at the image.

Type A draws uniformly from (A1, A2, A3, R); type B from (B1, B2), 50/50
as in the paper. Importing this module after coco_a3_plausible re-patches
data_loader.corrupt; call build_box_bank() before use.
"""

import os

import data_loader as dl
import coco_a3_plausible as a3

_bank = None      # bank[c] = [(x, y, w, h) normalized to image dims]


def build_box_bank(annotations_path):
    """Per-class bank of real ground-truth boxes in normalized coords."""
    global _bank
    import json
    samples, _ = dl.load_coco(annotations_path)
    # load_coco keeps no image dims; recover them from the raw json
    with open(os.path.expanduser(annotations_path)) as f:
        coco = json.load(f)
    dims = {im["file_name"]: (im["width"], im["height"])
            for im in coco["images"]}
    _bank = [[] for _ in range(dl.NUM_CLASSES)]
    for fname, ann in samples:
        w_img, h_img = dims[fname]
        for c, boxes in ann.items():
            for x, y, w, h in boxes:
                _bank[c].append((x / w_img, y / h_img,
                                 w / w_img, h / h_img))
    return _bank


def _bank_box(cls, rng, img_w, img_h):
    """A real box of class cls rescaled to this image; None if bank empty."""
    if not _bank or not _bank[cls]:
        return None
    x, y, w, h = _bank[cls][rng.randrange(len(_bank[cls]))]
    return [x * img_w, y * img_h, max(4.0, w * img_w), max(4.0, h * img_h)]


# Share of negatives that are A2, the only purely geometric error. The
# paper's mixture (50% type A over A1/A2/A3, 50% type B) leaves A2 at
# 1/6, and this generator's extra R subtype dilutes it further to 1/8 --
# about 6% of all training samples. Set this to raise the geometric
# share; None keeps the generator's own mixture.
A2_SHARE = None

# Extra bank-negative share: with probability B1_SHARE the negative is a
# B1 outright, before the normal 50/50 A/B split (which still produces
# its own B1s) -- same mechanism as A2_SHARE. B1_SHARE = 0.2 raises the
# effective B1 share from 25% to 40% of negatives. None keeps the
# generator's own mixture.
B1_SHARE = None

# Share of negatives drawn by the paper's own Section 5 generator
# (data_loader.PAPER_CORRUPT) instead of this module's, via corrupt_mixed.
# Either generator alone teaches half the taxonomy: the bank boxes here are
# what teach B detection, while the paper's generator keeps A1's multi-box
# deletions and uniform A3 swaps, which the test draws.
PAPER_MIX = 0.0


def corrupt_mixed(annotation, rng, img_w, img_h, subtype=None):
    """The paper's generator with probability PAPER_MIX, else this
    module's."""
    if rng.random() < PAPER_MIX:
        return dl.PAPER_CORRUPT(annotation, rng, img_w, img_h, subtype)
    return corrupt_imagehard(annotation, rng, img_w, img_h, subtype)


def corrupt_imagehard(annotation, rng, img_w, img_h, subtype=None):
    """corrupt() with label-space-ordinary negatives (see module doc)."""
    if subtype is None:
        if A2_SHARE is not None and rng.random() < A2_SHARE:
            subtype = "A2"
        elif B1_SHARE is not None and rng.random() < B1_SHARE:
            subtype = "B1"
        elif rng.random() < 0.5:
            subtype = rng.choice(("A1", "A2", "A3", "R"))
        else:
            subtype = rng.choice(("B1", "B2"))

    if subtype == "A1":
        out = dl.good(annotation, rng, img_w, img_h)
        flat = dl._flat(out)
        c, i = flat[rng.randrange(len(flat))]
        del out[c][i]
        if not out[c]:
            del out[c]
        return out, subtype

    if subtype == "R":
        out = dl.good(annotation, rng, img_w, img_h)
        flat = dl._flat(out)
        c, i = flat[rng.randrange(len(flat))]
        x, y, w, h = out[c][i]
        w = min(w, float(img_w))
        h = min(h, float(img_h))
        # 20 placements; if every one lands inside a region (near-image-
        # sized boxes) the last is kept, so that negative is a metric-good
        # relocation. A fallback would alter the data stream the shipped
        # checkpoint is trained on.
        for _ in range(20):
            box = [rng.uniform(0, img_w - w), rng.uniform(0, img_h - h),
                   w, h]
            if not dl._inside_any_region(box, annotation, img_w, img_h):
                break
        out[c][i] = box
        return out, subtype

    if subtype in ("B1", "B2"):
        out = dl.good(annotation, rng, img_w, img_h)
        present = sorted(annotation)
        for _ in range(rng.randint(1, 3)):
            if subtype == "B1":
                cls = rng.choice(present)
            else:
                candidates, scores = a3.plausible_classes(annotation)
                cls = (rng.choices(candidates, weights=scores, k=1)[0]
                       if candidates else rng.choice(present))
            for _ in range(20):
                box = (_bank_box(cls, rng, img_w, img_h)
                       or dl._random_box(rng, img_w, img_h))
                if not dl._inside_any_region(box, annotation, img_w, img_h):
                    break
            out.setdefault(cls, []).append(box)
        return out, subtype

    # A2 and A3 keep the a3-plausible behaviour exactly
    return a3.corrupt_a3_plausible(annotation, rng, img_w, img_h, subtype)


dl.corrupt = corrupt_imagehard
