"""Data pipeline: COCO annotations, label rasterisation, candidate
generation, the goodness metric and the raster head's targets.

A sample pairs an image with a candidate label. Good candidates are the
ground truth verbatim with probability p_exact (README item 3; run.sh
passes 0.5; validation, the refit and the test use 0), else every
ground-truth box re-drawn inside its uncertainty region — each coordinate
may move outward by up to REGION times that box's width/height (the
box-scaled reading, README interpretation 2), and never inward. The
paper's Section 5 generator (corrupt, also kept as PAPER_CORRUPT) starts
a bad candidate from a fresh good-style draw (the paper's error figures
show the untouched boxes region-jittered, not ground truth) and
introduces an error of one subtype:

    A1  erase one or multiple labels from uncertainty regions
    A2  distort a label so it falls outside its uncertainty region
    A3  swap classes of one or more labels from uncertainty regions
    B1  add new labels of ground-truth classes outside any uncertainty region
    B2  add new labels of classes absent from the ground truth

Errors are drawn 50% type A (subtypes uniform) and 50% type B (subtypes
uniform). Images are letterboxed to size x size with black padding; labels
are rendered as one channel per class with the box border set to 1.

Training-only sources, off by default: image-swap negatives (README item
11), detector candidates (detector_boxes, labelled by the whole-label
metric label_good; README item 12) and the augmentation set of the
author's thesis (build_augment, Listing 8.11). box_good is the per-box
form of the metric, which cell_targets turns into the raster head's
per-cell targets (README item 9); collate_sparse_planes / planes_to_dense
are the sparse plane transport.
"""

import json
import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

NUM_CLASSES = 80
REGION = 0.2

# Positive-sampler mixture; see good(). P_EXACT of positives are the ground
# truth verbatim (a good label by Equations 1-4 that a continuous draw
# would never produce); the remainder are uniform draws from the
# uncertainty region. The paper prescribes region membership
# (Equations 1-4), not a sampling density. P_NEAR, an optional
# squared-uniform near-ground-truth component, is off: it crowds the
# inner boundary, where A2 negatives begin only 2% away. run.sh overrides
# P_EXACT with --p_exact 0.5 (README item 3); evaluate.py, refit.py and
# the test-protocol validation set use 0.
P_EXACT = 0.25
P_NEAR = 0.0


def load_coco(annotations_path):
    """Parse a COCO instances file into per-image annotations.

    Images carrying any iscrowd annotation are excluded whole, as in the
    paper (Section 4.2: 5000 -> 4589 on the validation set) -- an unboxed
    crowd region contradicts the error supervision (its unlabeled objects
    pass as good while added boxes over it are marked bad). Degenerate
    boxes (non-positive extent) are dropped, and only images with at least
    one remaining box are kept, which also removes val2017's 48 images
    without annotations: the resulting pool has 4541 images, where the
    paper's count, taken before that step, is 4589.

    Returns (samples, names): samples is a list of
    (file_name, {class_index: [[x, y, w, h], ...]}) for images with at
    least one remaining box; names maps the contiguous class index
    (categories in ascending id order) to the class name.
    """
    with open(annotations_path) as f:
        coco = json.load(f)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    id2idx = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]
    files = {im["id"]: im["file_name"] for im in coco["images"]}
    crowd = {a["image_id"] for a in coco["annotations"] if a.get("iscrowd")}
    per_image = {}
    for a in coco["annotations"]:
        if (a["image_id"] in crowd
                or a["bbox"][2] <= 0 or a["bbox"][3] <= 0):
            continue
        entry = per_image.setdefault(a["image_id"], {})
        entry.setdefault(id2idx[a["category_id"]], []).append(list(a["bbox"]))
    return ([(files[i], ann) for i, ann in sorted(per_image.items())], names)


def get_sample(size, image, annotation, fill=0.0, border_px=1):
    """Render one (image, annotation) pair as model inputs.

    image: BGR uint8. annotation: {class_index: [[x, y, w, h], ...]} in
    original-image pixels. The image is resized keeping its aspect ratio
    and padded with black to size x size; each box border is drawn with
    value 1 in its class channel. The defaults are the paper rendering
    (1 px hollow border); fill > 0 additionally paints the interior at
    that value and border_px sets the border thickness (options outside
    the release recipe; train_coco_corr.py uses the defaults). Returns
    (cats, background), both CHW float32.
    """
    h, w = image.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    background = np.zeros((size, size, 3), np.float32)
    background[oy:oy + nh, ox:ox + nw] = cv2.resize(image, (nw, nh))
    cats = np.zeros((NUM_CLASSES, size, size), np.float32)
    for cls, boxes in annotation.items():
        for x, y, bw, bh in boxes:
            xa, xb = sorted((x, x + bw))
            ya, yb = sorted((y, y + bh))
            x1 = int(np.clip(xa * s, 0, nw - 1)) + ox
            y1 = int(np.clip(ya * s, 0, nh - 1)) + oy
            x2 = int(np.clip(xb * s, 0, nw - 1)) + ox
            y2 = int(np.clip(yb * s, 0, nh - 1)) + oy
            if x2 > x1 and y2 > y1:
                if fill > 0:
                    cv2.rectangle(cats[int(cls)], (x1, y1), (x2, y2),
                                  fill, -1)
                cv2.rectangle(cats[int(cls)], (x1, y1), (x2, y2), 1,
                              border_px)
    return cats, background.transpose(2, 0, 1)


def _ranges(box, img_w, img_h):
    """Per-coordinate uncertainty ranges (x1, y1, x2, y2) for one box,
    clipped to the image: the paper defines label coordinates on
    [0, width] x [0, height] (Equations 1-4), so a region never extends
    beyond the image.

    The allowance scales with the labelled box. Equations 1-4 read
    "width x 0.2" and the surrounding text glosses width and height as the
    input image's, but Figures 9-11 draw the outer region at 0.2 of the box
    extent -- in Figure 9's 640 x 480 image, 71 px beside a 359 px box,
    where the image reading predicts 128 px -- and that is the intended
    reading.
    """
    x, y, w, h = box
    return ((max(0.0, x - REGION * w), x),
            (max(0.0, y - REGION * h), y),
            (x + w, min(float(img_w), x + w + REGION * w)),
            (y + h, min(float(img_h), y + h + REGION * h)))


def _sample_in_region(box, rng, img_w, img_h, power=1.0):
    """Draw a box from the uncertainty region. Each coordinate's outward
    offset is u**power times its range: power 1 is the uniform draw, larger
    powers concentrate density near the ground truth (the inner endpoint)."""
    (a, b), (c, d), (e, f), (g, k) = _ranges(box, img_w, img_h)
    x1 = b - rng.random() ** power * (b - a)
    y1 = d - rng.random() ** power * (d - c)
    x2 = e + rng.random() ** power * (f - e)
    y2 = g + rng.random() ** power * (k - g)
    return [x1, y1, x2 - x1, y2 - y1]


def _copy(annotation):
    return {c: [list(b) for b in lst] for c, lst in annotation.items()}


def _flat(annotation):
    return [(c, i) for c, lst in annotation.items() for i in range(len(lst))]


def _inside_any_region(box, annotation, img_w, img_h):
    """True if box coordinate-fits the uncertainty region of any ground-truth
    box regardless of class: the paper places type-B additions "outside of
    any uncertainty region"."""
    x, y, w, h = box
    for lst in annotation.values():
        for gt in lst:
            (a, b), (c, d), (e, f), (g, k) = _ranges(gt, img_w, img_h)
            if (a <= x <= b and c <= y <= d
                    and e <= x + w <= f and g <= y + h <= k):
                return True
    return False


def good(annotation, rng, img_w, img_h, p_exact=P_EXACT, p_near=P_NEAR):
    """Good candidate: every box re-drawn inside its uncertainty region.

    With probability p_exact (README item 3) the ground truth is returned
    unchanged: it is
    a good label by Equations 1-4 -- the inner boundary of all four ranges
    -- but under a continuous draw it is a measure-zero event, so a correct
    annotation would never appear as a positive. The remainder are uniform
    draws over the uncertainty region (the paper prescribes membership,
    not density). p_near draws squared-uniform offsets dense near the
    inner boundary instead; it defaults to 0, because near-boundary
    positives collapse the margin to A2 negatives starting 2% away.
    """
    r = rng.random()
    if r < p_exact:
        return _copy(annotation)
    power = 2.0 if r < p_exact + p_near else 1.0
    return {c: [_sample_in_region(b, rng, img_w, img_h, power) for b in lst]
            for c, lst in annotation.items()}


def _random_box(rng, img_w, img_h):
    w = rng.uniform(0.05, 0.5) * img_w
    h = rng.uniform(0.05, 0.5) * img_h
    return [rng.uniform(0, img_w - w), rng.uniform(0, img_h - h), w, h]


def corrupt(annotation, rng, img_w, img_h, subtype=None):
    """Bad candidate: an error introduced to a freshly drawn good label.

    The base is drawn from the same mixture as positives: the paper's error
    figures (Figure 12) show the boxes untouched by the error region-jittered
    rather than at the ground truth, so the only signal separating the
    classes is the error itself. Uncertainty regions are always those of the
    ground truth. Returns (annotation, subtype). Subtype selection follows
    the paper: 50% type A with A1/A2/A3 uniform, 50% type B with B1/B2
    uniform; A1/A3 affect one or more labels and B1/B2 add 1-3 boxes, as in
    the paper's figures.
    """
    out = good(annotation, rng, img_w, img_h)
    flat = _flat(out)
    if subtype is None:
        if rng.random() < 0.5:
            subtype = rng.choice(("A1", "A2", "A3"))
        else:
            subtype = rng.choice(("B1", "B2"))

    if subtype == "A1":
        for c, i in sorted(rng.sample(flat, rng.randint(1, len(flat))),
                           key=lambda t: -t[1]):
            del out[c][i]
            if not out[c]:
                del out[c]
    elif subtype == "A2":
        c, i = flat[rng.randrange(len(flat))]
        x, y, w, h = out[c][i]
        coords = [x, y, x + w, y + h]
        gw, gh = annotation[c][i][2], annotation[c][i][3]
        r = _ranges(annotation[c][i], img_w, img_h)
        # The offence scales with the box, like the region it leaves.
        gdims = (gw, gh, gw, gh)
        idims = (img_w, img_h, img_w, img_h)
        for j in rng.sample(range(4), rng.randint(1, 4)):
            # The violation must stay inside the image (label coordinates
            # live on [0, width] x [0, height]): push past the outer region
            # edge only when there is room for a visible offense there,
            # otherwise past the inner edge, which always has room.
            off = rng.uniform(0.02, REGION) * gdims[j]
            lo, hi = r[j]
            if j < 2:
                outward = lo - off if lo - off >= 0.0 else None
                inward = hi + off
            else:
                outward = hi + off if hi + off <= idims[j] else None
                inward = lo - off
            pick = outward if (outward is not None
                               and rng.random() < 0.5) else inward
            coords[j] = min(max(pick, 0.0), float(idims[j]))
        x1, y1, x2, y2 = coords
        if x2 <= x1:
            x2 = min(x1 + 4, float(img_w))
            x1 = max(0.0, x2 - 4)
        if y2 <= y1:
            y2 = min(y1 + 4, float(img_h))
            y1 = max(0.0, y2 - 4)
        out[c][i] = [x1, y1, x2 - x1, y2 - y1]
    elif subtype == "A3":
        for c, i in sorted(rng.sample(flat, rng.randint(1, len(flat))),
                           key=lambda t: -t[1]):
            c2 = rng.choice([k for k in range(NUM_CLASSES) if k != c])
            out.setdefault(c2, []).append(out[c].pop(i))
            if not out[c]:
                del out[c]
    else:
        present = sorted(annotation)
        absent = [k for k in range(NUM_CLASSES) if k not in annotation]
        for _ in range(rng.randint(1, 3)):
            if subtype == "B1" or not absent:
                cls = rng.choice(present)
            else:
                cls = rng.choice(absent)
            for _ in range(20):
                box = _random_box(rng, img_w, img_h)
                if not _inside_any_region(box, annotation, img_w, img_h):
                    break
            out.setdefault(cls, []).append(box)
    return out, subtype


# Section 5's generator under a second name: coco_a3_plausible and
# coco_hard_negatives rebind data_loader.corrupt when imported, and the
# hard-negative mix (coco_hard_negatives.corrupt_mixed) draws from this one.
PAPER_CORRUPT = corrupt


def box_good(box, cls, annotation, img_w, img_h):
    """Per-box form of Equations 1-4: True if a ground-truth box of the
    same class has an uncertainty region containing box."""
    x, y, w, h = box
    for gt in annotation.get(cls, ()):
        (a, b), (c, d), (e, f), (g, k) = _ranges(gt, img_w, img_h)
        if (a <= x <= b and c <= y <= d
                and e <= x + w <= f and g <= y + h <= k):
            return True
    return False


def _in_region(box, gt, img_w, img_h):
    x, y, w, h = box
    (a, b), (c, d), (e, f), (g, k) = _ranges(gt, img_w, img_h)
    return (a <= x <= b and c <= y <= d
            and e <= x + w <= f and g <= y + h <= k)


def label_good(candidate, annotation, img_w, img_h):
    """Whole-label form of Equations 1-4, for candidates that no error
    generator produced (detector output, README item 12): good only if,
    class by class, the candidate has as many boxes as the ground truth
    and they pair one-to-one with the ground-truth boxes, each inside its
    own box's uncertainty region (a bipartite matching by augmenting
    paths). A missing box (A1), a box outside its region (A2), a wrong
    class (A3) and an added box (B1, B2) each break the pairing."""
    classes = ({c for c, b in candidate.items() if len(b)}
               | {c for c, b in annotation.items() if len(b)})
    for c in classes:
        boxes, gts = candidate.get(c, []), annotation.get(c, [])
        if len(boxes) != len(gts):
            return False
        adjacency = [[j for j, b in enumerate(boxes)
                      if _in_region(b, gt, img_w, img_h)] for gt in gts]
        match = [-1] * len(boxes)

        def augment(r, seen):
            for j in adjacency[r]:
                if j not in seen:
                    seen.add(j)
                    if match[j] < 0 or augment(match[j], seen):
                        match[j] = r
                        return True
            return False

        if not all(augment(r, set()) for r in range(len(gts))):
            return False
    return True


def build_augment():
    """The augmentation set of the author's thesis (Listing 8.11), applied
    to a share augment_p of the training samples (LabelQualityDataset;
    run.sh passes 0.1, one sample in ten) before the candidate is drawn.
    Its random draws are albumentations' own and are not seeded."""
    import albumentations as A
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2,
                      p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
                                   p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
                           rotate_limit=45, p=0.5),
        A.MotionBlur(blur_limit=7, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                             val_shift_limit=20, p=0.5),
        A.ChannelShuffle(p=0.5),
        A.OneOf([
            A.GaussianBlur(p=0.5),
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=7, p=0.1),
        ], p=0.5),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 80),
                        hole_width_range=(8, 80), p=0.5),
        A.CLAHE(p=0.5),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


# Detector candidates (README item 12): two open-source COCO detectors,
# boxes above DETECTOR_CONFIDENCE, loaded lazily once per process and run
# on the CPU; outputs are cached per (detector, image) within a process.
DETECTOR_CONFIDENCE = 0.9
_detectors = None
_detector_cache = {}


def detector_boxes(path, image_bgr, rng, class_names):
    """{class_index: [[x, y, w, h], ...]} in original pixels from one of
    the two detectors (SSDLite320- or FasterRCNN-MobileNetV3-320, drawn
    with rng), classes mapped to this pool's by name."""
    global _detectors
    import torch
    if _detectors is None:
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
            ssdlite320_mobilenet_v3_large,
            SSDLite320_MobileNet_V3_Large_Weights)
        w1 = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        w2 = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        _detectors = [(ssdlite320_mobilenet_v3_large(weights=w1).eval(), w1),
                      (fasterrcnn_mobilenet_v3_large_320_fpn(weights=w2)
                       .eval(), w2)]
    k = rng.randrange(len(_detectors))
    model, weights = _detectors[k]
    if (k, path) not in _detector_cache:
        rgb = torch.from_numpy(np.ascontiguousarray(
            image_bgr[..., ::-1])).permute(2, 0, 1)
        with torch.no_grad():
            out = model([weights.transforms()(rgb)])[0]
        _detector_cache[(k, path)] = (out["boxes"].numpy(),
                                      out["scores"].numpy(),
                                      out["labels"].numpy())
    boxes, scores, labels = _detector_cache[(k, path)]
    index = {n: i for i, n in enumerate(class_names)}
    out = {}
    for (x1, y1, x2, y2), s, lab in zip(boxes, scores, labels):
        name = weights.meta["categories"][int(lab)]
        if s > DETECTOR_CONFIDENCE and name in index and x2 > x1 and y2 > y1:
            out.setdefault(index[name], []).append(
                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    return out


def cell_targets(size, img_w, img_h, candidate, annotation, stride=8,
                 force_bad=False, class_annotation=None):
    """Training targets of the raster head (coco_corrnet raster_head,
    README item 9), at the size/stride grid:

    T (n_present, g, g) uint8, one map per class plane the sparse raster
    ships (same order): on every cell a claimed box's outline crosses,
    box_good of that box (the minimum where several cross), 255 elsewhere.
    The outline pixels are mapped exactly as get_sample draws them, so the
    claimed cells are the class plane max-pooled stride x stride.
    force_bad sets every claimed cell to 0 (an image-swap negative, README
    item 11: no box of the label is on the image shown).

    C (g, g) uint8: the class of the ground-truth box covering each cell
    (the smallest on top), NUM_CLASSES where none does -- what the image
    shows, never what the candidate claims; class_annotation replaces the
    ground truth here when the image shown is another one."""
    s = size / max(img_h, img_w)
    nw, nh = max(1, int(img_w * s)), max(1, int(img_h * s))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    g = size // stride
    present = sorted(c for c, b in candidate.items() if len(b))
    T = np.full((len(present), g, g), 255, np.uint8)
    for i, cls in enumerate(present):
        t = T[i]
        for box in candidate[cls]:
            x, y, bw, bh = box
            xa, xb = sorted((x, x + bw))
            ya, yb = sorted((y, y + bh))
            x1 = int(np.clip(xa * s, 0, nw - 1)) + ox
            y1 = int(np.clip(ya * s, 0, nh - 1)) + oy
            x2 = int(np.clip(xb * s, 0, nw - 1)) + ox
            y2 = int(np.clip(yb * s, 0, nh - 1)) + oy
            if not (x2 > x1 and y2 > y1):
                continue            # get_sample does not draw it either
            v = 0 if force_bad else int(box_good(box, cls, annotation,
                                                  img_w, img_h))
            x1, y1, x2, y2 = (q // stride for q in (x1, y1, x2, y2))
            for seg in (t[y1, x1:x2 + 1], t[y2, x1:x2 + 1],
                        t[y1:y2 + 1, x1], t[y1:y2 + 1, x2]):
                np.minimum(seg, v, out=seg)
    C = np.full((g, g), NUM_CLASSES, np.uint8)
    shown = annotation if class_annotation is None else class_annotation
    gts = sorted(((bw * bh, c, x, y, bw, bh)
                  for c, lst in shown.items()
                  for x, y, bw, bh in lst), reverse=True)
    for _, c, x, y, bw, bh in gts:
        x1 = int((np.clip(x * s, 0, nw - 1) + ox) / stride)
        y1 = int((np.clip(y * s, 0, nh - 1) + oy) / stride)
        x2 = int((np.clip((x + bw) * s, 0, nw - 1) + ox) / stride)
        y2 = int((np.clip((y + bh) * s, 0, nh - 1) + oy) / stride)
        C[y1:y2 + 1, x1:x2 + 1] = int(c)
    return T, C


class LabelQualityDataset(Dataset):
    """Balanced good/bad samples over a COCO split.

    Index parity selects the requested class: even indices yield a good
    candidate, odd indices a bad one -- an image-swap negative
    (swap_share), else a detector candidate (detector_share of the rest),
    else the active corrupt(). A detector candidate carries the label
    label_good gives it, so an odd index can be good. Draws are
    deterministic in (seed, epoch, index), apart from the augmentation's
    own draws; call set_epoch each epoch to regenerate samples per the
    paper's dynamic generation (its Figure 17 validation loss keeps
    improving across 250 epochs, which frozen draws cannot reproduce), or
    hold the epoch fixed for a fixed set (validation, the refit and the
    test draw fixed rounds).
    """

    def __init__(self, samples, images_dir, size=640, seed=0,
                 fill=0.0, border_px=1, gt_map_grid=0, gt_counts=False,
                 hflip=False, sparse_planes=False, cell_targets=False,
                 augment_p=0.0, detector_share=0.0, swap_share=0.0,
                 class_names=None):
        self.samples = samples
        self.images_dir = images_dir
        self.size = size
        self.seed = seed
        self.fill = fill
        self.border_px = border_px
        self.gt_map_grid = gt_map_grid
        self.gt_counts = gt_counts
        self.hflip = hflip
        # Training-only sources, all off by default (each consumes no rng
        # draw when off, so evaluation pools are unchanged):
        # augment_p      -- share of samples augmented with build_augment()
        #                   before the candidate is drawn
        # swap_share     -- share of negatives that pair a good label with
        #                   another pool image (image-swap negatives)
        # detector_share -- share of the remaining negatives taken from a
        #                   detector (detector_boxes), labelled by the
        #                   whole-label metric (label_good); needs
        #                   class_names
        self.augment_p = augment_p
        self.detector_share = detector_share
        self.swap_share = swap_share
        self.class_names = class_names
        if detector_share > 0 and class_names is None:
            raise ValueError("detector_share needs class_names")
        self._transform = build_augment() if augment_p > 0 else None
        # sparse_planes ships only the class channels that hold boxes
        # (with collate_sparse_planes / planes_to_dense); the dense
        # NUM_CLASSES x size x size float32 raster is ~99% zeros and
        # dominates worker transport. Transport-only: the scattered
        # batch is bit-identical to the dense one.
        self.sparse_planes = sparse_planes
        # cell_targets appends the raster head's training targets
        # (data_loader.cell_targets; collate with collate_cells)
        if cell_targets and (gt_map_grid or gt_counts or not sparse_planes):
            raise ValueError("cell_targets needs sparse_planes and no "
                             "gt_map_grid / gt_counts")
        self.cell_targets = cell_targets
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return 2 * len(self.samples)

    def __getitem__(self, index):
        file_name, annotation = self.samples[index // 2]
        is_good = index % 2 == 0
        rng = random.Random(self.seed * 1_000_003
                            + self.epoch * 7_776_146_593 + index)
        path = os.path.join(self.images_dir, file_name)
        image = cv2.imread(path)
        h, w = image.shape[:2]
        swap = (not is_good and self.swap_share > 0
                and rng.random() < self.swap_share)
        detected = None
        if (not is_good and not swap and self.detector_share > 0
                and rng.random() < self.detector_share):
            detected = detector_boxes(path, image, rng, self.class_names)
        if self._transform is not None and rng.random() < self.augment_p:
            image, annotation, detected = self._augment(image, annotation,
                                                        detected)
            h, w = image.shape[:2]
        label, shown = int(is_good), None
        if is_good:
            candidate = good(annotation, rng, w, h)
        elif swap:
            # a good label of this image over another pool image: bad as a
            # pair, and invisible to the label geometry (README item 11)
            candidate = good(annotation, rng, w, h)
            image, shown = self._other_image(index // 2, rng, w, h)
        elif detected is not None:
            candidate = detected
            label = int(label_good(detected, annotation, w, h))
        else:
            candidate, _ = corrupt(annotation, rng, w, h)
        cats, background = get_sample(self.size, image, candidate,
                                      self.fill, self.border_px)
        extras = []
        if self.gt_map_grid:
            # Ground-truth class map at the fusion grid (an option
            # train_coco_corr.py does not use): positions inside a GT box
            # carry its class index, everything else is 255 (ignored).
            # Built from the GROUND TRUTH, never the candidate — a
            # corrupted candidate's planes carry wrong classes by design.
            g = self.gt_map_grid
            s = self.size / max(h, w)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            ox, oy = (self.size - nw) // 2, (self.size - nh) // 2
            gt_map = np.full((g, g), 255, np.uint8)
            f = g / self.size
            for cls, boxes in annotation.items():
                for x, y, bw, bh in boxes:
                    x1 = int((np.clip(x * s, 0, nw - 1) + ox) * f)
                    y1 = int((np.clip(y * s, 0, nh - 1) + oy) * f)
                    x2 = int((np.clip((x + bw) * s, 0, nw - 1) + ox) * f)
                    y2 = int((np.clip((y + bh) * s, 0, nh - 1) + oy) * f)
                    gt_map[y1:max(y1 + 1, y2 + 1),
                           x1:max(x1 + 1, x2 + 1)] = int(cls)
            extras.append(gt_map)
        if self.gt_counts:
            # Ground-truth per-class box counts (an option
            # train_coco_corr.py does not use), also from the GROUND
            # TRUTH: what the image contains, not what the candidate claims.
            counts = np.zeros(NUM_CLASSES, np.float32)
            for cls, boxes in annotation.items():
                counts[int(cls)] = len(boxes)
            extras.append(counts)
        flipped = self.hflip and rng.random() < 0.5
        if flipped:
            # Horizontal flip augmentation: the task is left-right
            # symmetric, so flip image, label planes, and the gt class
            # map together. Drawn from the same per-index rng AFTER the
            # candidate, so candidate draws match a non-flip run exactly.
            cats = np.ascontiguousarray(cats[..., ::-1])
            background = np.ascontiguousarray(background[..., ::-1])
            extras = [np.ascontiguousarray(e[..., ::-1])
                      if e.ndim == 2 else e for e in extras]
        if self.cell_targets:
            extras = list(cell_targets(self.size, w, h, candidate,
                                       annotation, force_bad=swap,
                                       class_annotation=shown))
            if flipped:
                # size is a multiple of the stride, so pixel x -> size-1-x
                # sends cell j to g-1-j exactly
                extras = [np.ascontiguousarray(e[..., ::-1])
                          for e in extras]
        if self.sparse_planes:
            present = np.array(sorted(c for c, b in candidate.items()
                                      if len(b)), np.int64)
            return ((present, np.ascontiguousarray(cats[present])),
                    background, label, *extras)
        return (cats, background, label, *extras)

    def _other_image(self, own, rng, w, h):
        """Another pool image stretched to this sample's w x h, and its
        ground truth in that frame."""
        j = own
        while j == own and len(self.samples) > 1:
            j = rng.randrange(len(self.samples))
        file_name, ann = self.samples[j]
        other = cv2.imread(os.path.join(self.images_dir, file_name))
        oh, ow = other.shape[:2]
        sx, sy = w / ow, h / oh
        shown = {c: [[x * sx, y * sy, bw * sx, bh * sy]
                     for x, y, bw, bh in lst] for c, lst in ann.items()}
        return cv2.resize(other, (w, h), interpolation=cv2.INTER_AREA), shown

    def _augment(self, image, annotation, detected):
        """build_augment() over the image with the ground truth and any
        detector boxes transformed alongside; the unaugmented sample is
        kept if no ground-truth box survives."""
        h, w = image.shape[:2]
        flat, tags = [], []
        for src, boxes in (("gt", annotation), ("det", detected or {})):
            for c, lst in boxes.items():
                for x, y, bw, bh in lst:
                    x1, y1 = max(0.0, x), max(0.0, y)
                    x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
                    if x2 - x1 >= 1 and y2 - y1 >= 1:
                        flat.append([x1, y1, x2, y2])
                        tags.append((src, c))
        try:
            out = self._transform(image=image, bboxes=flat,
                                  labels=list(range(len(flat))))
        except Exception:
            return image, annotation, detected
        new_ann, new_det = {}, ({} if detected is not None else None)
        for (x1, y1, x2, y2), k in zip(out["bboxes"], out["labels"]):
            src, c = tags[int(k)]
            dest = new_ann if src == "gt" else new_det
            dest.setdefault(c, []).append([float(x1), float(y1),
                                           float(x2 - x1), float(y2 - y1)])
        if not new_ann:
            return image, annotation, detected
        return out["image"], new_ann, new_det


def collate_sparse_planes(batch):
    """Collate for sparse_planes samples: backgrounds and labels stack;
    per-sample plane stacks concatenate with (sample, class) indices."""
    import torch as _t
    sample_idx, cls_idx, chunks = [], [], []
    for i, b in enumerate(batch):
        present, planes = b[0]
        sample_idx += [i] * len(present)
        cls_idx.append(present)
        chunks.append(planes)
    size = batch[0][1].shape[-1]
    planes = _t.from_numpy(np.concatenate(chunks, 0)) if chunks else \
        _t.zeros(0, size, size, dtype=_t.float32)
    idx = (_t.tensor(sample_idx, dtype=_t.int64),
           _t.from_numpy(np.concatenate(cls_idx)) if cls_idx else
           _t.zeros(0, dtype=_t.int64))
    background = _t.from_numpy(np.stack([b[1] for b in batch]))
    rest = [_t.tensor(np.array([b[k] for b in batch]))
            for k in range(2, len(batch[0]))]
    return ((idx, planes), background, *rest)


def collate_cells(batch):
    """Collate for cell_targets samples: collate_sparse_planes for the
    first three items, then the per-plane targets concatenated in the
    planes' order (P, g, g) and the class maps stacked (B, g, g)."""
    import torch as _t
    head = collate_sparse_planes([b[:3] for b in batch])
    T = _t.from_numpy(np.concatenate([b[3] for b in batch], 0))
    C = _t.from_numpy(np.stack([b[4] for b in batch]))
    rest = [_t.tensor(np.array([b[k] for b in batch]))
            for k in range(5, len(batch[0]))]
    return (*head, T, C, *rest)


def planes_to_dense(cats_pack, background, device):
    """Scatter a collate_sparse_planes cats pack to the dense
    (B, NUM_CLASSES, S, S) raster on `device`."""
    (sample_idx, cls_idx), planes = cats_pack
    B, S = background.shape[0], background.shape[-1]
    out = __import__("torch").zeros(B, NUM_CLASSES, S, S, device=device)
    if planes.shape[0]:
        out[sample_idx, cls_idx] = planes.to(device)
    return out


def cv2_single_thread_worker(_worker_id):
    """DataLoader worker_init_fn: one cv2 thread per worker. cv2's
    default per-process pool times a large worker count oversubscribes
    the host; drawing output is unaffected."""
    cv2.setNumThreads(0)
