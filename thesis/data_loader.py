"""Data pipeline: Pascal VOC 2012 pools, uncertainty regions, candidates.

A sample pairs an image with a candidate label for the whole image (all of
its bounding boxes). The uncertainty region of a ground-truth box corner is
a two-sided band around it: each coordinate may move outward by alpha times
the box dimension and inward by beta times the box dimension (Chapter 7,
alpha = beta = 0.05). Good candidates draw every corner inside its region.
In the thesis generator -- the evaluation draw of evaluate.py, analysis.py,
train.py's validation set and refit.py -- bad candidates are produced with
a 10% chance by an open-source object detector (SSDLite320-MobileNetV3 or
FasterRCNN-MobileNetV3-320, confidence above 0.9, COCO classes mapped to
VOC) and with a 90% chance by corrupting a good draw with the error types
of Table 8.2:

    1  erase one or multiple label(s)
    2  swap classes of labels
    3  translate one or multiple label(s)
    4  resize one or multiple label(s)
    5  crop one or multiple label(s)
    6  combine two or more boxes into a single larger box
    7  split a single box into multiple smaller boxes
    8  apply small random translations to box coordinates

with equal probability of applying one error type or a combination with
repetition of ten. Every candidate is labeled good or bad by the
uncertainty-region metric (Equation 7.1).

Training candidates (Data with augment=True, as train.py builds it) run
the Listing 8.11 augmentations on one sample in ten; half of the good candidates are the
ground truth verbatim; of the negatives, 10% are image-swap negatives,
10% of the rest come from the detectors, and the remainder is split
evenly between the thesis generator and image-hard negatives.

Images are resized aspect-preserving to fit 224 x 224 and centered on black
padding, scaled to [0, 1] and normalized with the ImageNet statistics
(Equation 8.1). Labels are rendered as 20 one-per-class 224 x 224 planes
with the box border and both diagonals set to 1.

Implementation details vs the thesis (see README):
- good_label takes p_exact_gt: with that probability the positive is the
  ground-truth annotation verbatim (README item 1; train-time only, the
  evaluation draw keeps the thesis draw).
- Images are converted BGR -> RGB and normalized with the standard-order
  ImageNet statistics, so the pretrained image branch receives the
  convention its weights expect (part of README item 2).
- corrupt can report which error types it applied (README item 3,
  reporting only).
- cell_targets builds the per-cell head's training targets (README item 4).
- build_context and corrupt_mixed make the image-hard negatives (README
  item 7).
- Data's swap_share makes the image-swap negatives (README item 8).
- fixed_rows seeds the fixed validation set (README item 9) and the refit
  population (README item 6).
"""

import copy
import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']
MAPPING = {n: i for i, n in enumerate(VOC_CLASSES)}
NUM_CLASSES = len(VOC_CLASSES)

RES = 224
ALPHA = 0.05
BETA = 0.05
DETECTOR_SHARE = 0.1
DETECTOR_CONFIDENCE = 0.9
# Probability that a requested good candidate is the ground truth verbatim
# (README item 1). The thesis draw yields the exact annotation only by
# coincidence, with a probability that vanishes as an image gains boxes,
# although it is the label a user actually submits. Applied at train time
# only; the evaluation draw (evaluate.py, validation, refit.py) keeps the
# thesis value 0.
P_EXACT_GT = 0.5

COCO_TO_VOC = {'motorcycle': 'motorbike', 'airplane': 'aeroplane',
               'couch': 'sofa', 'potted plant': 'pottedplant',
               'dining table': 'diningtable', 'tv': 'tvmonitor'}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def ann_to_img(ann_path):
    base = os.path.basename(ann_path).split(".")[0]
    return "/".join(ann_path.split("/")[:-2]) + "/data/" + base + ".jpg"


def resize_keep_aspect(image, size=RES):
    h, w = image.shape[:2]
    s = size / max(h, w)
    return cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_NEAREST)


def center_pad(plane, size, channels):
    if channels == 1:
        out = np.zeros((size, size), dtype=np.float32)
    else:
        out = np.zeros((size, size, channels), dtype=np.float32)
    oy = (size - plane.shape[0]) // 2
    ox = (size - plane.shape[1]) // 2
    out[oy:oy + plane.shape[0], ox:ox + plane.shape[1]] = plane
    return out


def draw_box(plane, x1, y1, x2, y2):
    cv2.rectangle(plane, (x1, y1), (x2, y2), 1, 1)
    cv2.line(plane, (x1, y1), (x2, y2), 1, 1)
    cv2.line(plane, (x1, y2), (x2, y1), 1, 1)


def render_planes(bbs, plane_h, plane_w, size=RES):
    planes = np.zeros((NUM_CLASSES, size, size), dtype=np.float32)
    for key, boxes in bbs.items():
        plane = np.zeros((plane_h, plane_w), dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            draw_box(plane, int(x1), int(y1), int(x2), int(y2))
        planes[int(key)] = center_pad(plane, size, 1)
    return planes


def box_good(box, cls, regions):
    """Per-box form of Equation 7.1: True if a same-class uncertainty
    region holds this box inside its bands. The label-level metric adds
    the one-to-one matching (is_label_negative); per box, band membership
    is what a single box can be judged on."""
    return any(_in_band(box, r) for r in regions[int(cls)])


def cell_targets(bbs, regions, plane_h, plane_w, size=RES, stride=8,
                 force_bad=False, shown=None):
    """Training targets of the label-plane head (neural_network.py
    raster_head, README item 4), on the size/stride grid:

    T (NUM_CLASSES, g, g) uint8: on every cell a claimed box's drawing
    covers, box_good of that box (the minimum where several cover it),
    255 elsewhere. The drawing is rendered exactly as render_planes draws
    it -- outline and both diagonals, centre-padded -- so the claimed
    cells are the class plane max-pooled stride x stride. force_bad sets
    every claimed cell to 0 (an image-swap negative, README item 8: no box
    of the label is on the image shown).

    C (g, g) uint8: the class of the ground-truth box covering each cell
    (the smallest on top), NUM_CLASSES where none does -- what the image
    shows, never what the candidate claims; shown ({key: [[x1, y1, x2,
    y2]]}) replaces the ground truth here when the image is another one."""
    g = size // stride
    T = np.full((NUM_CLASSES, g, g), 255, np.uint8)
    for key, boxes in bbs.items():
        for box in boxes:
            plane = np.zeros((plane_h, plane_w), dtype=np.float32)
            draw_box(plane, *[int(c) for c in box])
            cells = center_pad(plane, size, 1).reshape(
                g, stride, g, stride).max(axis=(1, 3)) > 0
            t = T[int(key)]
            v = np.uint8(0 if force_bad else box_good(box, key, regions))
            np.minimum(t, np.where(cells, v, np.uint8(255)), out=t)
    C = np.full((g, g), NUM_CLASSES, np.uint8)
    oy, ox = (size - plane_h) // 2, (size - plane_w) // 2
    if shown is None:
        gts = [(r['gt'], key) for key in range(NUM_CLASSES)
               for r in regions[key]]
    else:
        gts = [(tuple(b), key) for key, lst in shown.items() for b in lst]
    gts = sorted(gts,
                 key=lambda t: -(t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))
    for (x1, y1, x2, y2), key in gts:
        C[(int(y1) + oy) // stride:(int(y2) + oy) // stride + 1,
          (int(x1) + ox) // stride:(int(x2) + ox) // stride + 1] = key
    return T, C


def normalize_image(image, size=RES):
    img = center_pad(image / 255, size, 3)
    return ((img - MEAN) / STD).transpose(2, 0, 1)


def build_regions(bbs, img_w, img_h):
    """Per-coordinate uncertainty bands (outer bound first, inner second)."""
    regions = {key: [] for key in range(NUM_CLASSES)}
    for key, boxes in bbs.items():
        for x1, y1, x2, y2 in boxes:
            w = x2 - x1
            h = y2 - y1
            cx = lambda v: int(np.clip(v, 0, img_w - 1))
            cy = lambda v: int(np.clip(v, 0, img_h - 1))
            regions[int(key)].append(
                {'x1': (cx(x1 - w * ALPHA), cx(x1 + w * BETA)),
                 'y1': (cy(y1 - h * ALPHA), cy(y1 + h * BETA)),
                 'x2': (cx(x2 + w * ALPHA), cx(x2 - w * BETA)),
                 'y2': (cy(y2 + h * ALPHA), cy(y2 - h * BETA)),
                 'gt': (cx(x1), cy(y1), cx(x2), cy(y2))})
    return regions


def _in_band(box, region):
    x1, y1, x2, y2 = box
    return (region['x1'][0] <= x1 <= region['x1'][1] and
            region['y1'][0] <= y1 <= region['y1'][1] and
            region['x2'][1] <= x2 <= region['x2'][0] and
            region['y2'][1] <= y2 <= region['y2'][0])


def _has_perfect_matching(adjacency, n_boxes):
    """One-to-one region-to-box assignment via augmenting paths (Kuhn)."""
    match = [-1] * n_boxes

    def augment(r, seen):
        for j in adjacency[r]:
            if j in seen:
                continue
            seen.add(j)
            if match[j] < 0 or augment(match[j], seen):
                match[j] = r
                return True
        return False

    return all(augment(r, set()) for r in range(len(adjacency)))


def is_label_negative(generated_bbs, uncertainty_regions):
    """Equation 7.1: bad unless every region holds exactly one in-band box.

    "Exactly one" is decided by a perfect matching between regions and
    same-class boxes: a candidate is good only if a one-to-one assignment
    pairs every region with its own in-band box.
    """
    for key in range(NUM_CLASSES):
        boxes = generated_bbs.get(key, [])
        regs = uncertainty_regions[key]
        if len(boxes) != len(regs):
            return True
        if not regs:
            continue
        adjacency = [[j for j, b in enumerate(boxes) if _in_band(b, r)]
                     for r in regs]
        if any(not a for a in adjacency):
            return True
        if not _has_perfect_matching(adjacency, len(boxes)):
            return True
    return False


def _draw_coord(bounds):
    lo, hi = min(bounds), max(bounds)
    return random.randint(lo, hi)


def good_label(regions, p_exact_gt=0.0):
    """Every box corner drawn inside its uncertainty region.

    With probability p_exact_gt the ground-truth annotation is returned
    verbatim (README item 1). Under the thesis draw every corner is
    redrawn inside its band, so the exact annotation — the one label a
    user actually submits — comes up only by coincidence, with a
    probability that vanishes as an image gains boxes. The ground truth is
    trivially inside every band, so the metric label is unchanged.
    """
    if p_exact_gt > 0 and random.random() < p_exact_gt:
        bbs = {key: [] for key in range(NUM_CLASSES)}
        ok = True
        for key in range(NUM_CLASSES):
            for region in regions[key]:
                x1, y1, x2, y2 = region['gt']
                if x2 > x1 and y2 > y1:
                    bbs[key].append([x1, y1, x2, y2])
                else:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return bbs
    for _ in range(25):
        bbs = {key: [] for key in range(NUM_CLASSES)}
        ok = True
        for key in range(NUM_CLASSES):
            for region in regions[key]:
                x1 = _draw_coord(region['x1'])
                y1 = _draw_coord(region['y1'])
                x2 = _draw_coord(region['x2'])
                y2 = _draw_coord(region['y2'])
                if x2 > x1 and y2 > y1:
                    bbs[key].append([x1, y1, x2, y2])
                else:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return bbs
    return {key: [] for key in range(NUM_CLASSES)}


def erase_labels(bbs):
    for key in bbs:
        bbs[key] = [b for b in bbs[key] if random.randint(0, 9) != 0]
    return bbs


def swap_classes(bbs):
    out = copy.deepcopy(bbs)
    class_keys = list(out.keys())
    for key in class_keys:
        to_swap = [b for b in out[key] if random.randint(0, 1) == 0]
        for box in to_swap:
            out[key].remove(box)
            new_key = key
            while new_key == key:
                new_key = random.choice(class_keys)
            out[new_key].append(box)
    return out


def translate(bbs, img_w, img_h):
    for key in bbs:
        for i, (x1, y1, x2, y2) in enumerate(bbs[key]):
            if random.randint(0, 1) != 0:
                continue
            w, h = x2 - x1, y2 - y1
            dx = random.randint(-max(1, w), max(1, w))
            dy = random.randint(-max(1, h), max(1, h))
            nx1 = int(np.clip(x1 + dx, 0, img_w - 1))
            ny1 = int(np.clip(y1 + dy, 0, img_h - 1))
            nx2 = int(np.clip(x2 + dx, 0, img_w - 1))
            ny2 = int(np.clip(y2 + dy, 0, img_h - 1))
            if nx2 > nx1 and ny2 > ny1:
                bbs[key][i] = [nx1, ny1, nx2, ny2]
    return bbs


def resize_boxes(bbs, img_w, img_h, min_scale=0.04, max_scale=25.0):
    for key in bbs:
        for i, (x1, y1, x2, y2) in enumerate(bbs[key]):
            if random.randint(0, 1) != 0:
                continue
            w, h = x2 - x1, y2 - y1
            s = random.uniform(min_scale, max_scale)
            cx, cy = x1 + w / 2, y1 + h / 2
            nx1 = int(np.clip(cx - w * s / 2, 0, img_w - 1))
            ny1 = int(np.clip(cy - h * s / 2, 0, img_h - 1))
            nx2 = int(np.clip(cx + w * s / 2, 0, img_w - 1))
            ny2 = int(np.clip(cy + h * s / 2, 0, img_h - 1))
            if nx2 > nx1 and ny2 > ny1:
                bbs[key][i] = [nx1, ny1, nx2, ny2]
    return bbs


def crop_boxes(bbs, img_w, img_h, crop_fraction=0.5):
    for key in bbs:
        for i, (x1, y1, x2, y2) in enumerate(bbs[key]):
            if random.randint(0, 1) != 0:
                continue
            w, h = x2 - x1, y2 - y1
            cx1 = x1 + w * (1 - crop_fraction) * random.uniform(0, 1)
            cy1 = y1 + h * (1 - crop_fraction) * random.uniform(0, 1)
            cx2 = x2 - w * (1 - crop_fraction) * random.uniform(0, 1)
            cy2 = y2 - h * (1 - crop_fraction) * random.uniform(0, 1)
            if cx1 > cx2:
                cx1, cx2 = cx2, cx1
            if cy1 > cy2:
                cy1, cy2 = cy2, cy1
            nx1 = int(np.clip(cx1, 0, img_w - 1))
            ny1 = int(np.clip(cy1, 0, img_h - 1))
            nx2 = int(np.clip(cx2, 0, img_w - 1))
            ny2 = int(np.clip(cy2, 0, img_h - 1))
            if nx2 > nx1 and ny2 > ny1:
                bbs[key][i] = [nx1, ny1, nx2, ny2]
    return bbs


def combine_boxes(bbs, max_combinations=5):
    out = {}
    for key, boxes in bbs.items():
        boxes = [list(b) for b in boxes]
        original = [list(b) for b in boxes]
        combined = []
        remaining = max_combinations
        while len(boxes) >= 2 and remaining > 0:
            if len(boxes) == 2 or remaining == 1:
                n = 2
            else:
                n = random.randint(2, min(len(boxes), max_combinations))
            picked = random.sample(range(len(boxes)), n)
            box = [int(c) for c in boxes[picked[0]]]
            for idx in picked[1:]:
                x1, y1, x2, y2 = box
                a1, b1, a2, b2 = [int(c) for c in boxes[idx]]
                box = [min(x1, a1), min(y1, b1), max(x2, a2), max(y2, b2)]
            combined.append(box)
            remaining -= 1
            for idx in sorted(picked, reverse=True):
                del boxes[idx]
        combined.extend(boxes)
        if not combined:
            combined = original
        out[key] = [[int(c) for c in b] for b in combined]
    return out


def split_boxes(bbs, img_w, img_h, max_splits=5):
    out = {}
    for key, boxes in bbs.items():
        split = []
        for box in boxes:
            if random.randint(0, 1) != 0:
                split.append(box)
                continue
            x1, y1, x2, y2 = [int(c) for c in box]
            w, h = x2 - x1, y2 - y1
            for _ in range(random.randint(1, max_splits)):
                sw = random.uniform(0.3, 0.7) * w
                sh = random.uniform(0.3, 0.7) * h
                sx1 = random.uniform(x1, x2 - sw)
                sy1 = random.uniform(y1, y2 - sh)
                nx1 = int(np.clip(sx1, 0, img_w - 1))
                ny1 = int(np.clip(sy1, 0, img_h - 1))
                nx2 = int(np.clip(sx1 + sw, 0, img_w - 1))
                ny2 = int(np.clip(sy1 + sh, 0, img_h - 1))
                if nx2 > nx1 and ny2 > ny1:
                    split.append([nx1, ny1, nx2, ny2])
                else:
                    split.append(box)
        out[key] = split
    return out


def jitter_boxes(bbs, img_w, img_h, max_jitter=112):
    for key in bbs:
        for i, (x1, y1, x2, y2) in enumerate(bbs[key]):
            if random.randint(0, 1) != 0:
                continue
            dx = random.randint(-max_jitter, max_jitter)
            dy = random.randint(-max_jitter, max_jitter)
            nx1 = int(np.clip(x1 + dx, 0, img_w - 1))
            ny1 = int(np.clip(y1 + dy, 0, img_h - 1))
            nx2 = int(np.clip(x2 + dx, 0, img_w - 1))
            ny2 = int(np.clip(y2 + dy, 0, img_h - 1))
            if nx2 > nx1 and ny2 > ny1:
                bbs[key][i] = [nx1, ny1, nx2, ny2]
    return bbs


def corrupt(regions, img_w, img_h, return_ops=False, force_ops=None):
    """Bad candidate: corrupt a good draw until the metric rejects it.

    return_ops / force_ops exist for the per-error-type report (README
    item 3) and change nothing when left at their defaults.
    """
    ops = ('erase', 'swap', 'translate', 'resize', 'crop', 'combine', 'split',
           'jitter')
    for _ in range(25):
        bbs = good_label(regions)
        if force_ops is not None:
            choices = list(force_ops)
        elif random.randint(0, 1) == 0:
            choices = [random.choice(ops)]
        else:
            choices = random.choices(ops, k=10)
        for op in choices:
            if op == 'erase':
                bbs = erase_labels(bbs)
            elif op == 'swap':
                bbs = swap_classes(bbs)
            elif op == 'translate':
                bbs = translate(bbs, img_w, img_h)
            elif op == 'resize':
                bbs = resize_boxes(bbs, img_w, img_h)
            elif op == 'crop':
                bbs = crop_boxes(bbs, img_w, img_h)
            elif op == 'combine':
                bbs = combine_boxes(bbs)
            elif op == 'split':
                bbs = split_boxes(bbs, img_w, img_h)
            elif op == 'jitter':
                bbs = jitter_boxes(bbs, img_w, img_h)
        valid = all(x2 > x1 and y2 > y1
                    for boxes in bbs.values() for x1, y1, x2, y2 in boxes)
        if valid and is_label_negative(bbs, regions):
            return (bbs, choices) if return_ops else bbs
    empty = {key: [] for key in range(NUM_CLASSES)}
    return (empty, []) if return_ops else empty


# ------------------------------------------------ image-hard negatives --
# (README item 7.) Bad labels whose geometry is statistically ordinary, so
# only the image can give them away: exactly one box deleted, one box
# relabelled to an absent class that co-occurs with the present ones, one
# box relocated at its own size to a random place, or 1-3 real boxes of a
# present or plausible absent class added from a per-class bank of
# ground-truth boxes. The bank and co-occurrence table come from the
# training pool (build_context) and travel on the dataset (Data context).

def build_context(files):
    """(bank, cooc) over annotation files: bank[c] holds every class-c
    ground-truth box as (x, y, w, h) normalised by the image size; cooc
    [c][d] is the share of images with class c that also show class d."""
    from PIL import Image
    bank = [[] for _ in range(NUM_CLASSES)]
    cooc = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for f in files:
        packed = np.load(f, allow_pickle=True).item()
        with Image.open(ann_to_img(f)) as im:
            img_w, img_h = im.size
        present = [int(c) for c, b in packed.items()
                   if int(c) < NUM_CLASSES and len(b)]
        for c in present:
            for x, y, w, h in packed[c]:
                if w > 0 and h > 0:
                    bank[c].append((x / img_w, y / img_h, w / img_w,
                                    h / img_h))
            for d in present:
                if d != c:
                    cooc[c][d] += 1
    cooc /= np.maximum(cooc.sum(1, keepdims=True), 1.0)
    return bank, cooc


def _flat(bbs):
    return [(k, i) for k, lst in bbs.items() for i in range(len(lst))]


def _plausible_absent(present, cooc):
    absent = [k for k in range(NUM_CLASSES) if k not in present]
    if not absent:
        return random.choice(sorted(present))
    scores = [sum(cooc[c][k] for c in present) for k in absent]
    if sum(scores) <= 0:
        return random.choice(absent)
    return random.choices(absent, weights=scores, k=1)[0]


def _clip(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return [float(np.clip(x1, 0, img_w - 1)), float(np.clip(y1, 0, img_h - 1)),
            float(np.clip(x2, 0, img_w - 1)), float(np.clip(y2, 0, img_h - 1))]


def ih_delete_one(bbs, ctx, img_w, img_h):
    flat = _flat(bbs)
    if len(flat) < 2:
        return None
    key, i = random.choice(flat)
    del bbs[key][i]
    return bbs


def ih_swap_plausible(bbs, ctx, img_w, img_h):
    flat = _flat(bbs)
    if not flat:
        return None
    key, i = random.choice(flat)
    new_key = _plausible_absent({k for k, _ in flat}, ctx[1])
    if new_key == key:
        return None
    bbs[new_key].append(bbs[key].pop(i))
    return bbs


def ih_relocate(bbs, ctx, img_w, img_h):
    flat = _flat(bbs)
    if not flat:
        return None
    key, i = random.choice(flat)
    x1, y1, x2, y2 = bbs[key][i]
    w, h = min(x2 - x1, img_w - 1), min(y2 - y1, img_h - 1)
    nx, ny = random.uniform(0, img_w - w), random.uniform(0, img_h - h)
    box = _clip([nx, ny, nx + w, ny + h], img_w, img_h)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    bbs[key][i] = box
    return bbs


def ih_add_bank(bbs, ctx, img_w, img_h, plausible_absent=False):
    flat = _flat(bbs)
    if not flat:
        return None
    present = {k for k, _ in flat}
    bank = ctx[0]
    for _ in range(random.randint(1, 3)):
        cls = (_plausible_absent(present, ctx[1]) if plausible_absent
               else random.choice(sorted(present)))
        if not bank[cls]:
            continue
        x, y, w, h = random.choice(bank[cls])
        box = _clip([x * img_w, y * img_h, (x + w) * img_w, (y + h) * img_h],
                    img_w, img_h)
        if box[2] > box[0] and box[3] > box[1]:
            bbs[cls].append(box)
    return bbs


IH_OPS = (ih_delete_one, ih_swap_plausible, ih_relocate,
          lambda b, c, w, h: ih_add_bank(b, c, w, h, False),
          lambda b, c, w, h: ih_add_bank(b, c, w, h, True))


def corrupt_mixed(regions, img_w, img_h, ctx, ih_share):
    """A bad candidate: with probability ih_share an image-hard negative
    (README item 7: a good draw with one IH_OPS error, kept only if the
    metric rejects it; 30 tries, then the thesis generator), otherwise the
    thesis generator (corrupt)."""
    if ctx is not None and random.random() < ih_share:
        for _ in range(30):
            bbs = random.choice(IH_OPS)(good_label(regions), ctx,
                                        img_w, img_h)
            if (bbs is not None and any(bbs.values())
                    and all(x2 > x1 and y2 > y1 for lst in bbs.values()
                            for x1, y1, x2, y2 in lst)
                    and is_label_negative(bbs, regions)):
                return bbs
    return corrupt(regions, img_w, img_h)


def build_augment():
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


class Data(Dataset):
    """Balanced good/bad candidates over annotation files.

    paths is a list of (annotation_path, requested_label) pairs; requested
    label 1 asks for a good draw, 0 for a bad one. The returned label is
    re-derived with the uncertainty-region metric (image-swap negatives
    excepted: always bad) — in particular a detector-generated candidate
    may turn out good. A row may carry a
    third element, a seed: the draw is then reseeded with it, so the row
    always yields the same candidate (fixed_rows: the validation set and the
    refit).

    Training-only sources (augment=True; evaluation draws are unchanged):
    one sample in ten passes through the Listing 8.11 augmentations, and
    detector candidates and the ground truth pass through them together;
    swap_share of the negatives pair a good label with another pool image
    stretched to this image's size (image-swap negatives, always bad,
    README item 8); ih_share of the generated negatives are image-hard
    (corrupt_mixed, README item 7; context from build_context).
    """

    def __init__(self, paths, augment=False, detector_share=DETECTOR_SHARE,
                 p_exact_gt=0.0, cell_targets=False, ih_share=0.0,
                 context=None, swap_share=0.0):
        self.paths = paths
        self.augment = augment
        self.detector_share = detector_share
        self.p_exact_gt = p_exact_gt
        self.ih_share = ih_share
        self.context = context
        self.swap_share = swap_share
        if (ih_share > 0 or swap_share > 0) and not augment:
            raise ValueError("ih_share / swap_share are training sources "
                             "(augment=True)")
        # cell_targets appends the label-plane head's training targets
        # (data_loader.cell_targets)
        self.cell_targets = cell_targets
        self.label_cache = {}
        self.pred_cache = {}
        self._models = None
        self._transform = None
        if augment:
            self._transform = build_augment()

    def __len__(self):
        return len(self.paths)

    def _detectors(self):
        if self._models is None:
            from torchvision.models.detection import (
                fasterrcnn_mobilenet_v3_large_320_fpn,
                FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                ssdlite320_mobilenet_v3_large,
                SSDLite320_MobileNet_V3_Large_Weights)
            w1 = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            m1 = ssdlite320_mobilenet_v3_large(weights=w1).eval()
            w2 = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            m2 = fasterrcnn_mobilenet_v3_large_320_fpn(weights=w2).eval()
            self._models = [(m1, w1), (m2, w2)]
        return self._models

    def _detector_bbs(self, img_path, old_w, old_h, new_w, new_h):
        import torch
        from torchvision.io.image import read_image
        model, weights = random.choice(self._detectors())
        cache_key = (model.__class__.__name__, img_path)
        if cache_key in self.pred_cache:
            prediction = self.pred_cache[cache_key]
        else:
            with torch.no_grad():
                batch = [weights.transforms()(read_image(img_path))]
                out = model(batch)[0]
            prediction = {'scores': out['scores'].numpy(),
                          'boxes': out['boxes'].numpy(),
                          'labels': out['labels'].numpy().astype(np.int32)}
            self.pred_cache[cache_key] = prediction
        names = [weights.meta["categories"][i] for i in prediction['labels']]
        bbs = {key: [] for key in range(NUM_CLASSES)}
        for idx, box in enumerate(prediction['boxes']):
            name = COCO_TO_VOC.get(names[idx], names[idx])
            if name not in MAPPING:
                continue
            if float(prediction['scores'][idx]) <= DETECTOR_CONFIDENCE:
                continue
            x1 = int(np.clip(box[0] / old_w * new_w, 0, new_w - 1))
            y1 = int(np.clip(box[1] / old_h * new_h, 0, new_h - 1))
            x2 = int(np.clip(box[2] / old_w * new_w, 0, new_w - 1))
            y2 = int(np.clip(box[3] / old_h * new_h, 0, new_h - 1))
            if x2 > x1 and y2 > y1:
                bbs[MAPPING[name]].append([x1, y1, x2, y2])
        return bbs

    def _other_image(self, own, new_w, new_h):
        """Another pool image stretched to new_w x new_h, and its ground
        truth in that frame (the image of an image-swap negative, README
        item 8)."""
        other = own
        while other == own and len(self.paths) > 1:
            other = random.choice(self.paths)[0]
        image = cv2.cvtColor(cv2.imread(ann_to_img(other)), cv2.COLOR_BGR2RGB)
        oh, ow = image.shape[:2]
        sx, sy = new_w / ow, new_h / oh
        shown = {}
        for key, segs in np.load(other, allow_pickle=True).item().items():
            if int(key) < NUM_CLASSES:
                shown[int(key)] = [_clip([x * sx, y * sy, (x + w) * sx,
                                          (y + h) * sy], new_w, new_h)
                                   for x, y, w, h in segs]
        return cv2.resize(image, (new_w, new_h),
                          interpolation=cv2.INTER_AREA), shown

    def __getitem__(self, index):
        row = self.paths[index]
        path, requested = row[0], row[1]
        if len(row) > 2:
            random.seed(row[2])
        if path in self.label_cache:
            packed = self.label_cache[path]
        else:
            packed = np.load(path, allow_pickle=True).item()
            self.label_cache[path] = packed

        img_path = ann_to_img(path)
        image = cv2.imread(img_path)
        # BGR -> RGB so the ImageNet-pretrained image branch receives the channel
        # convention its weights expect (part of README item 2).
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        old_h, old_w = image.shape[:2]
        image = resize_keep_aspect(image)
        new_h, new_w = image.shape[:2]

        boxes = {key: [] for key in range(NUM_CLASSES)}
        for key, segs in packed.items():
            if int(key) >= NUM_CLASSES:
                continue
            for x, y, w, h in segs:
                x1 = np.clip(x / old_w * new_w, 0, new_w - 1)
                y1 = np.clip(y / old_h * new_h, 0, new_h - 1)
                x2 = np.clip((x + w) / old_w * new_w, 0, new_w - 1)
                y2 = np.clip((y + h) / old_h * new_h, 0, new_h - 1)
                boxes[int(key)].append([x1, y1, x2, y2])

        if not self.augment:
            # the evaluation draw (evaluate.py, analysis.py, refit.py, validation)
            regions = build_regions(boxes, new_w, new_h)
            if requested == 1:
                bbs = good_label(regions, self.p_exact_gt)
            elif random.random() < self.detector_share:
                bbs = self._detector_bbs(img_path, old_w, old_h, new_w, new_h)
            else:
                bbs = corrupt(regions, new_w, new_h)
            label = 0 if is_label_negative(bbs, regions) else 1
            planes = render_planes(bbs, new_h, new_w)
            img = normalize_image(image)
            if self.cell_targets:
                return (img, planes, label,
                        *cell_targets(bbs, regions, new_h, new_w))
            return img, planes, label

        # training: the candidate's source is chosen before augmenting, so
        # detector boxes pass through the same transform as the ground
        # truth; the Listing 8.11 augmentations run on one sample in ten
        swap = (requested == 0 and self.swap_share > 0
                and random.random() < self.swap_share)
        detected = None
        if (requested == 0 and not swap
                and random.random() < self.detector_share):
            detected = self._detector_bbs(img_path, old_w, old_h,
                                          new_w, new_h)
        if random.randint(0, 9) == 0:
            flat = [(b, ('gt', key)) for key, lst in boxes.items()
                    for b in lst]
            if detected is not None:
                flat += [(b, ('det', key)) for key, lst in detected.items()
                         for b in lst]
            try:
                transformed = self._transform(
                    image=image, bboxes=[b for b, _ in flat],
                    labels=list(range(len(flat))))
                new_boxes = {key: [] for key in range(NUM_CLASSES)}
                new_det = ({key: [] for key in range(NUM_CLASSES)}
                           if detected is not None else None)
                for box, k in zip(transformed['bboxes'],
                                  transformed['labels']):
                    src, key = flat[int(k)][1]
                    (new_boxes if src == 'gt' else new_det)[key].append(
                        [float(c) for c in box])
                if any(new_boxes.values()):
                    image = transformed['image']
                    new_h, new_w = image.shape[:2]
                    boxes, detected = new_boxes, new_det
            except Exception:
                pass

        regions = build_regions(boxes, new_w, new_h)
        shown = None
        if requested == 1:
            bbs = good_label(regions, self.p_exact_gt)
        elif swap:
            bbs = good_label(regions, self.p_exact_gt)
            image, shown = self._other_image(path, new_w, new_h)
        elif detected is not None:
            bbs = detected
        else:
            bbs = corrupt_mixed(regions, new_w, new_h, self.context,
                                self.ih_share)

        label = 0 if swap or is_label_negative(bbs, regions) else 1
        planes = render_planes(bbs, new_h, new_w)
        img = normalize_image(image)
        if self.cell_targets:
            return (img, planes, label,
                    *cell_targets(bbs, regions, new_h, new_w,
                                  force_bad=swap, shown=shown))
        return img, planes, label


def fixed_rows(files, rounds, seed, offset=0):
    """Both requests of every file over draw rounds offset .. offset +
    rounds - 1, each row seeded so it always yields the same candidate:
    the fixed validation set (train.py, rounds 0-3, README item 9) and the
    refit population (refit.py, rounds 4-7, README item 6) of one split
    never share a draw."""
    rows = []
    for r in range(offset, offset + rounds):
        for k, f in enumerate(files):
            for req in (0, 1):
                rows.append((f, req, seed * 1_000_003 + r * 7_776_146_593
                             + 2 * k + req))
    return rows

