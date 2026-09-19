"""Unified data layer: candidates, metric, rendering, one negative mixer.

Dataset-agnostic; everything dataset-specific (class list, input
resolution, fusion grid, pool loading, detector name mapping) comes from
`dataset.py` (selected by QA_DATASET=coco|voc).

A label is good iff every uncertainty region (outward ALPHA, inward BETA
x the box dimension) holds exactly one box of the right class, decided
by a perfect region-to-box matching. Every candidate -- drawn, corrupted
or detector-produced -- is re-labelled by that metric.

Negatives come from one mixer with four sources:
  image-swap    (share swap_share, drawn first; 0 by default, 0.1 in
                run_qa.sh): a metric-good label paired with a different
                pool image, labelled bad;
  detector      (share DETECTOR_SHARE of the non-swap negatives):
                SSDLite320-MobileNetV3-Large /
                FasterRCNN-MobileNetV3-Large-320-FPN output above
                confidence 0.9, classes mapped per dataset;
  image-hard    (IMAGEHARD_SHARE of the rest): single deletion, context-
                plausible class swap, off-region relocation, and real
                boxes of present (B1) / plausible-absent (B2) classes
                drawn from a bank built over the training pool;
  severity ops  (the remainder): coordinate and structural corruptions at
                a controlled severity (the `balanced` mix).
With probability COMPOSE_PROB (the compose share) an image-hard or
severity negative stacks 1-3 extra errors.

Rendering: per-class planes with a filled interior (0.25), a border of cv2
thickness 2 (3 px wide)
and both diagonals; images are RGB, ImageNet-normalised. Augmentation
(horizontal flip, mild affine, photometric; p 0.5) transforms image,
ground truth and detector boxes together before candidates are drawn.
"""

import dataset as _dataset


import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

CLASSES = _dataset.CLASSES
MAPPING = {n: i for i, n in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

RES = _dataset.RES
ALPHA = 0.05
BETA = 0.05
MIN_BOX_PX = 2                 # boxes thinner than this cannot be rasterised

DETECTOR_SHARE = 0.1
DETECTOR_CONFIDENCE = 0.9
DETECTOR_NAME_MAP = _dataset.DETECTOR_NAME_MAP

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FILL = 0.25                    # interior value of a rendered box
# Data ships planes as uint8 x PLANE_SCALE (lossless for {0, FILL, 1});
# consumers divide back after moving to the device. A float32 640-px
# batch is ~130 MB per sample and exhausts DataLoader shared memory at
# large batch sizes.
PLANE_SCALE = 4
BORDER_PX = 2                  # cv2 line thickness of the border (renders 3 px wide)

# Share of good candidates that are the ground truth verbatim rather than a
# jittered draw inside the band.
P_EXACT_GT = 0.5

# Share of jittered good candidates whose corners are drawn inside the
# quarter of their band nearest either edge instead of uniformly: near-boundary legal labels,
# so the model learns where the boundary is rather than a fuzzy prototype
# of "aligned". 0 keeps the plain uniform draw (and consumes no rng); the
# release recipe uses 0.25 (run_qa.sh).
P_EDGE = 0.0

# Share of erase errors that drop the smallest boxes rather than a uniform
# sample.
P_ERASE_SMALL = 0.7

# How far beyond the uncertainty band an error is pushed, relative to the box
# dimension. The band edge sits at ALPHA = 0.05, so 0.01 falls just outside.
REL_EXTRA = {'subtle': (0.01, 0.05),
             'moderate': (0.05, 0.25),
             'gross': (0.25, 1.50),
             'catastrophic': (1.50, 12.0)}

# How many boxes and coordinates one coordinate error may touch. A subtle
# error is always a single coordinate of a single box, which keeps it a
# near-boundary case.
SEVERITY_SPAN = {'subtle': 1, 'moderate': 2, 'gross': 4, 'catastrophic': 6}

# `weights` picks the coordinate-error magnitude; `p_struct` is the share of
# negatives produced by a structural error instead (relabel / erase / add /
# split / merge); `p_compose` stacks extra structural errors on top.
#
# Each setting defines its own negative distribution, so figures obtained
# under one are not comparable with figures obtained under another.
SEVERITY_MIX = {
    # weighted towards the decision boundary
    'balanced': {'weights': {'subtle': 0.30, 'moderate': 0.35,
                             'gross': 0.25, 'catastrophic': 0.10},
                 'p_struct': 0.30, 'p_compose': 0.10},
    # shifted towards larger errors
    'coarse': {'weights': {'subtle': 0.02, 'moderate': 0.08,
                           'gross': 0.30, 'catastrophic': 0.60},
               'p_struct': 0.46, 'p_compose': 0.55},
}

SEVERITY_CODES = {'good': 0, 'subtle': 1, 'moderate': 2, 'gross': 3,
                  'catastrophic': 4, 'count': 5, 'detector': 6}
CODE_NAMES = {v: k for k, v in SEVERITY_CODES.items()}


# --------------------------------------------------------------- geometry --
def resize_keep_aspect(image, size=RES):
    h, w = image.shape[:2]
    s = size / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=interp)


def center_pad(plane, size, channels):
    shape = (size, size) if channels == 1 else (size, size, channels)
    out = np.zeros(shape, dtype=np.float32)
    oy = (size - plane.shape[0]) // 2
    ox = (size - plane.shape[1]) // 2
    out[oy:oy + plane.shape[0], ox:ox + plane.shape[1]] = plane
    return out


def center_pad_uint8(image, size=RES):
    """Letterboxed raw BGR uint8 HWC -- normalisation happens on the
    GPU in to_dense_batch, matching normalize_image exactly."""
    h, w = image.shape[:2]
    out = np.zeros((size, size, 3), np.uint8)
    oy, ox = (size - h) // 2, (size - w) // 2
    out[oy:oy + h, ox:ox + w] = image
    return out


def collate_sparse(batch):
    """Collate for sparse_planes samples: images stack; per-sample
    plane stacks concatenate with an index vector mapping each plane to
    (sample, class)."""
    import torch as _t
    imgs = _t.from_numpy(np.stack([b[0] for b in batch]))
    sample_idx, cls_idx, chunks = [], [], []
    for i, b in enumerate(batch):
        present, sparse = b[1]
        sample_idx += [i] * len(present)
        cls_idx.append(present)
        chunks.append(sparse)
    planes = _t.from_numpy(np.concatenate(chunks, 0)) if chunks else \
        _t.zeros(0, batch[0][0].shape[0], batch[0][0].shape[1],
                 dtype=_t.uint8)
    idx = (_t.tensor(sample_idx, dtype=_t.int64),
           _t.from_numpy(np.concatenate(cls_idx)) if cls_idx else
           _t.zeros(0, dtype=_t.int64))
    rest = []
    for k in range(2, len(batch[0])):
        vals = [b[k] for b in batch]
        if isinstance(vals[0], np.ndarray) and vals[0].ndim == 2 \
                and vals[0].shape[-1] == BOX_FEATS:
            # per-box rows: concatenate across samples with the sample
            # index prepended, matching roi_align's batch-index column
            rows = [np.concatenate(
                [np.full((v.shape[0], 1), i, np.float32), v], 1)
                for i, v in enumerate(vals)]
            rest.append(_t.from_numpy(np.concatenate(rows, 0)))
        else:
            rest.append(_t.tensor(np.array(vals)))
    return (imgs, planes, idx, *rest)


_IMAGENET_MEAN_T = None
_IMAGENET_STD_T = None


def to_dense_batch(imgs, planes, idx, device):
    """GPU side of the sparse path: scatter shipped planes into the
    dense (B, NUM_CLASSES, H, W) raster and normalise the raw images --
    numerically identical to render_planes + normalize_image."""
    import torch as _t
    global _IMAGENET_MEAN_T, _IMAGENET_STD_T
    if _IMAGENET_MEAN_T is None or _IMAGENET_MEAN_T.device != device:
        _IMAGENET_MEAN_T = _t.tensor(IMAGENET_MEAN, device=device) \
            .view(1, 3, 1, 1)
        _IMAGENET_STD_T = _t.tensor(IMAGENET_STD, device=device) \
            .view(1, 3, 1, 1)
    B, H, W = imgs.shape[0], imgs.shape[1], imgs.shape[2]
    dense = _t.zeros(B, NUM_CLASSES, H, W, device=device)
    if planes.shape[0]:
        dense[idx[0].to(device), idx[1].to(device)] = \
            planes.to(device).float() / PLANE_SCALE
    # BGR uint8 HWC -> RGB normalised CHW, as normalize_image
    img = imgs.to(device).float().flip(-1).permute(0, 3, 1, 2) / 255.0
    img = (img - _IMAGENET_MEAN_T) / _IMAGENET_STD_T
    return img, dense


def normalize_image(image, size=RES):
    """image: BGR uint8 (OpenCV order) -> normalised CHW float32 in RGB order."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = center_pad(rgb.astype(np.float32) / 255.0, size, 3)
    return np.ascontiguousarray(
        ((img - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))


def render_plane(boxes, plane_h, plane_w, size=RES):
    """One class plane: filled interior, thick border, both diagonals.

    Fills for every box are drawn before any border, so an overlapping
    box cannot erase a neighbour's border.
    """
    plane = np.zeros((plane_h, plane_w), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(plane, (int(x1), int(y1)), (int(x2), int(y2)),
                      FILL, -1)
    for x1, y1, x2, y2 in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(plane, p1, p2, 1.0, BORDER_PX)
        cv2.line(plane, p1, p2, 1.0, 1)
        cv2.line(plane, (int(x1), int(y2)), (int(x2), int(y1)), 1.0, 1)
    return center_pad(plane, size, 1)


def render_planes(bbs, plane_h, plane_w, size=RES):
    """One plane per class (render_plane per non-empty class)."""
    planes = np.zeros((NUM_CLASSES, size, size), dtype=np.float32)
    for key, boxes in bbs.items():
        if boxes:
            planes[int(key)] = render_plane(boxes, plane_h, plane_w, size)
    return planes


def cv2_single_thread_worker(_worker_id):
    """DataLoader worker_init_fn: one cv2 thread per worker. cv2's
    default per-process pool times a large worker count oversubscribes
    the host; drawing output is unaffected."""
    cv2.setNumThreads(0)


def build_regions(bbs, img_w, img_h):
    """Per-coordinate uncertainty bands, plus the integer ground-truth box.

    The stored `gt` is used as a guaranteed-valid fallback when a random draw
    would collapse a very small box, so `good_label` can never fail.
    """
    regions = {key: [] for key in range(NUM_CLASSES)}
    for key, boxes in bbs.items():
        for x1, y1, x2, y2 in boxes:
            w, h = x2 - x1, y2 - y1
            cx = lambda v: int(np.clip(v, 0, img_w - 1))   # noqa: E731
            cy = lambda v: int(np.clip(v, 0, img_h - 1))   # noqa: E731
            regions[int(key)].append({
                'x1': (cx(x1 - w * ALPHA), cx(x1 + w * BETA)),
                'y1': (cy(y1 - h * ALPHA), cy(y1 + h * BETA)),
                'x2': (cx(x2 + w * ALPHA), cx(x2 - w * BETA)),
                'y2': (cy(y2 + h * ALPHA), cy(y2 - h * BETA)),
                'gt': (cx(x1), cy(y1), cx(x2), cy(y2)),
                'wh': (max(1.0, float(w)), max(1.0, float(h))),
            })
    return regions


def _widen(box, img_w, img_h):
    """Grow a sub-pixel box to MIN_BOX_PX, keeping it inside the frame."""
    x1, y1, x2, y2 = box
    if x2 - x1 < MIN_BOX_PX:
        cx = (x1 + x2) / 2.0
        x1, x2 = cx - MIN_BOX_PX / 2.0, cx + MIN_BOX_PX / 2.0
    if y2 - y1 < MIN_BOX_PX:
        cy = (y1 + y2) / 2.0
        y1, y2 = cy - MIN_BOX_PX / 2.0, cy + MIN_BOX_PX / 2.0
    x1 = max(0.0, min(x1, max(0.0, img_w - 1 - MIN_BOX_PX)))
    y1 = max(0.0, min(y1, max(0.0, img_h - 1 - MIN_BOX_PX)))
    return [x1, y1, min(float(img_w - 1), x1 + MIN_BOX_PX),
            min(float(img_h - 1), y1 + MIN_BOX_PX)]


def boxes_from_packed(packed, old_w, old_h, new_w, new_h):
    """Ground-truth boxes in resized coordinates.

    Boxes thinner than MIN_BOX_PX cannot be rasterised and are dropped. If
    dropping them would leave the image with no boxes at all they are widened
    instead, so the ground truth is never empty and no candidate becomes an
    all-zero raster.
    """
    scaled = []
    for key, segs in packed.items():
        if int(key) >= NUM_CLASSES:
            continue
        for x, y, w, h in segs:
            scaled.append((int(key), [
                float(np.clip(x / old_w * new_w, 0, new_w - 1)),
                float(np.clip(y / old_h * new_h, 0, new_h - 1)),
                float(np.clip((x + w) / old_w * new_w, 0, new_w - 1)),
                float(np.clip((y + h) / old_h * new_h, 0, new_h - 1))]))
    kept = [(k, b) for k, b in scaled
            if b[2] - b[0] >= MIN_BOX_PX and b[3] - b[1] >= MIN_BOX_PX]
    if not kept:
        kept = [(k, _widen(b, new_w, new_h)) for k, b in scaled]
    boxes = {key: [] for key in range(NUM_CLASSES)}
    for key, box in kept:
        boxes[key].append(box)
    return boxes


def _in_band(box, region):
    x1, y1, x2, y2 = box
    for value, key in ((x1, 'x1'), (y1, 'y1'), (x2, 'x2'), (y2, 'y2')):
        lo, hi = min(region[key]), max(region[key])
        if not lo <= value <= hi:
            return False
    return True


def _has_perfect_matching(adjacency, n_right):
    """Kuhn's algorithm: can every region be matched to a distinct box?"""
    match = [-1] * n_right

    def augment(u, seen):
        for v in adjacency[u]:
            if not seen[v]:
                seen[v] = True
                if match[v] == -1 or augment(match[v], seen):
                    match[v] = u
                    return True
        return False

    for u in range(len(adjacency)):
        if not augment(u, [False] * n_right):
            return False
    return True


def is_label_negative(generated_bbs, uncertainty_regions):
    """Equation 7.1: bad unless every region holds exactly one in-band box."""
    for key in range(NUM_CLASSES):
        boxes = generated_bbs.get(key, [])
        regions = uncertainty_regions[key]
        if len(boxes) != len(regions):
            return True
        if not regions:
            continue
        adjacency = [[j for j, b in enumerate(boxes) if _in_band(b, r)]
                     for r in regions]
        if any(not a for a in adjacency):
            return True
        if not _has_perfect_matching(adjacency, len(boxes)):
            return True
    return False


# ------------------------------------------------------- per-box targets --
# Row layout of the per-box supervision the Data class ships when per_box
# is on: letterboxed RoI coordinates, the claimed class, and the targets
# for the verification head (collate_sparse prepends the sample index).
BOX_FEATS = 8      # x1, y1, x2, y2, cls, ok, margin, margin_valid


def _max_matching(adjacency, n_right):
    """Kuhn over every region, no early exit: match[box] -> region or -1."""
    match = [-1] * n_right

    def augment(u, seen):
        for v in adjacency[u]:
            if not seen[v]:
                seen[v] = True
                if match[v] == -1 or augment(match[v], seen):
                    match[v] = u
                    return True
        return False

    for u in range(len(adjacency)):
        augment(u, [False] * n_right)
    return match


def _box_margin(box, region):
    """Signed distance to the band boundary in band-half-width units:
    +1 at the band centre, 0 exactly on the boundary, negative outside;
    clipped to [-5, 1]."""
    w, h = region['wh']
    worst = None
    for value, key, dim in ((box[0], 'x1', w), (box[1], 'y1', h),
                            (box[2], 'x2', w), (box[3], 'y2', h)):
        lo, hi = min(region[key]), max(region[key])
        m = min(value - lo, hi - value) / max(ALPHA * dim, 1e-6)
        worst = m if worst is None else min(worst, m)
    return float(np.clip(worst, -5.0, 1.0))


def per_box_targets(generated_bbs, uncertainty_regions):
    """Supervision for the per-box verification head, aligned with
    _flat(generated_bbs): parallel lists (ok, margin, margin_valid).

    ok: the box is paired with a region of its class by a maximum
    in-band matching -- a duplicate claim on an already-taken region or
    a box no region accepts gets 0 regardless of the per-class counts,
    so credit lands on the offending box alone. margin: signed band
    margin against the matched region (the nearest region of the class
    when unmatched); masked when the class has no regions at all.
    """
    ok, margin, valid = [], [], []
    for key in range(NUM_CLASSES):
        boxes = generated_bbs.get(key, [])
        if not boxes:
            continue
        regions = uncertainty_regions[key]
        adjacency = [[j for j, b in enumerate(boxes) if _in_band(b, r)]
                     for r in regions]
        match = _max_matching(adjacency, len(boxes))
        for j, box in enumerate(boxes):
            if not regions:
                ok.append(0.0)
                margin.append(0.0)
                valid.append(0.0)
                continue
            region = regions[match[j]] if match[j] >= 0 else \
                max(regions, key=lambda r: _box_margin(box, r))
            ok.append(1.0 if match[j] >= 0 else 0.0)
            margin.append(_box_margin(box, region))
            valid.append(1.0)
    return ok, margin, valid


# ------------------------------------------------------------- generation --
def _edge_coord(rng, lo, hi):
    """A coordinate inside the quarter of its band nearest either edge."""
    q = max(0, (hi - lo) // 4)
    return rng.randint(lo, lo + q) if rng.random() < 0.5 \
        else rng.randint(hi - q, hi)


def good_label(regions, rng, p_exact=P_EXACT_GT, p_edge=P_EDGE):
    """Every box corner drawn inside its uncertainty region.

    With probability `p_exact` the ground truth is returned verbatim, so
    perfectly aligned labels -- the kind a careful annotator produces -- are
    represented alongside jittered draws. Including them keeps the model from
    scoring perfectly aligned labels below jittered ones, so both are
    sampled. Each jittered box is additionally drawn near the band boundary
    with probability `p_edge` (see P_EDGE; 0 reproduces the plain uniform
    draw exactly, consuming no rng).

    Falls back to the ground-truth box, which always lies inside its own band,
    if random draws would collapse the box, so this never returns an empty
    label.
    """
    if rng.random() < p_exact:
        return {key: [list(r['gt']) for r in regions[key]]
                for key in range(NUM_CLASSES)}
    bbs = {key: [] for key in range(NUM_CLASSES)}
    for key in range(NUM_CLASSES):
        for region in regions[key]:
            edge = p_edge > 0 and rng.random() < p_edge
            box = None
            for _ in range(12):
                if edge:
                    x1 = _edge_coord(rng, *sorted(region['x1']))
                    y1 = _edge_coord(rng, *sorted(region['y1']))
                    x2 = _edge_coord(rng, *sorted(region['x2']))
                    y2 = _edge_coord(rng, *sorted(region['y2']))
                else:
                    x1 = rng.randint(*sorted(region['x1']))
                    y1 = rng.randint(*sorted(region['y1']))
                    x2 = rng.randint(*sorted(region['x2']))
                    y2 = rng.randint(*sorted(region['y2']))
                if x2 > x1 and y2 > y1:
                    box = [x1, y1, x2, y2]
                    break
            bbs[key].append(box if box is not None else list(region['gt']))
    return bbs


def _flat(bbs):
    return [(k, i) for k in range(NUM_CLASSES) for i in range(len(bbs[k]))]


def _total(bbs):
    return sum(len(v) for v in bbs.values())


def _valid(bbs):
    return all(x2 > x1 and y2 > y1
               for boxes in bbs.values() for x1, y1, x2, y2 in boxes)


def _extra(rng, severity, dim):
    lo, hi = REL_EXTRA[severity]
    return rng.uniform(lo, hi) * dim


def _clip_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return [int(np.clip(x1, 0, img_w - 1)), int(np.clip(y1, 0, img_h - 1)),
            int(np.clip(x2, 0, img_w - 1)), int(np.clip(y2, 0, img_h - 1))]


def op_perturb(bbs, regions, rng, severity, img_w, img_h):
    """Push individual coordinates just outside their band (error types 4/5)."""
    flat = _flat(bbs)
    if not flat:
        return None
    limit = SEVERITY_SPAN[severity]
    n_boxes = 1 if severity == 'subtle' else rng.randint(1, min(len(flat), limit))
    n_coords = 1 if severity == 'subtle' else rng.randint(1, min(4, limit))
    for key, idx in rng.sample(flat, n_boxes):
        region = regions[key][idx]
        box = list(bbs[key][idx])
        w, h = region['wh']
        for j in rng.sample(range(4), n_coords):
            name = ('x1', 'y1', 'x2', 'y2')[j]
            lo, hi = min(region[name]), max(region[name])
            step = _extra(rng, severity, w if j in (0, 2) else h)
            box[j] = (lo - step) if rng.random() < 0.5 else (hi + step)
        box = _clip_box(box, img_w, img_h)
        if box[2] > box[0] and box[3] > box[1]:
            bbs[key][idx] = box
    return bbs


def op_translate(bbs, regions, rng, severity, img_w, img_h):
    """Move whole boxes off their objects (error types 3 and 8)."""
    flat = _flat(bbs)
    if not flat:
        return None
    n = 1 if severity == 'subtle' else rng.randint(1, min(len(flat), 3))
    for key, idx in rng.sample(flat, n):
        w, h = regions[key][idx]['wh']
        x1, y1, x2, y2 = bbs[key][idx]
        dx = (ALPHA * w + _extra(rng, severity, w)) * rng.choice((-1, 1))
        dy = (ALPHA * h + _extra(rng, severity, h)) * rng.choice((-1, 1))
        box = _clip_box([x1 + dx, y1 + dy, x2 + dx, y2 + dy], img_w, img_h)
        if box[2] > box[0] and box[3] > box[1]:
            bbs[key][idx] = box
    return bbs


def op_scale(bbs, regions, rng, severity, img_w, img_h):
    """Grow or shrink boxes about their centre (error type 4; shrinking also covers type 5)."""
    flat = _flat(bbs)
    if not flat:
        return None
    n = 1 if severity == 'subtle' else rng.randint(1, min(len(flat), 3))
    for key, idx in rng.sample(flat, n):
        w, h = regions[key][idx]['wh']
        x1, y1, x2, y2 = bbs[key][idx]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # a coordinate moves by s/2 of the dimension; the band edge is ALPHA
        s = 2 * (ALPHA + _extra(rng, severity, 1.0))
        factor = (1 + s) if rng.random() < 0.5 else 1.0 / (1 + s)
        nw, nh = max(2.0, w * factor), max(2.0, h * factor)
        box = _clip_box([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2],
                        img_w, img_h)
        if box[2] > box[0] and box[3] > box[1]:
            bbs[key][idx] = box
    return bbs


def op_swap_class(bbs, rng, img_w, img_h):
    """Relabel a box (error type 2).  Always violates the per-class counts."""
    flat = _flat(bbs)
    if not flat:
        return None
    key, idx = rng.choice(flat)
    new_key = rng.choice([k for k in range(NUM_CLASSES) if k != key])
    bbs[new_key].append(bbs[key].pop(idx))
    return bbs


def op_erase(bbs, rng, img_w, img_h, small_bias=P_ERASE_SMALL):
    """Drop boxes (error type 1) -- never all of them.

    With probability `small_bias` the smallest boxes are dropped rather than a
    uniform sample.
    """
    flat = _flat(bbs)
    if len(flat) < 2:
        return None
    n = rng.randint(1, len(flat) - 1)
    if rng.random() < small_bias:
        def area(t):
            x1, y1, x2, y2 = bbs[t[0]][t[1]]
            return (x2 - x1) * (y2 - y1)
        chosen = sorted(flat, key=area)[:n]
    else:
        chosen = rng.sample(flat, n)
    for key, idx in sorted(chosen, key=lambda t: -t[1]):
        del bbs[key][idx]
    return bbs


def op_add(bbs, rng, img_w, img_h):
    """Invent a spurious box."""
    key = rng.randrange(NUM_CLASSES)
    w = rng.uniform(0.08, 0.5) * img_w
    h = rng.uniform(0.08, 0.5) * img_h
    x1 = rng.uniform(0, max(1.0, img_w - w))
    y1 = rng.uniform(0, max(1.0, img_h - h))
    box = _clip_box([x1, y1, x1 + w, y1 + h], img_w, img_h)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    bbs[key].append(box)
    return bbs


def op_split(bbs, rng, img_w, img_h):
    """Split one box into two or three smaller ones (error type 7)."""
    flat = _flat(bbs)
    if not flat:
        return None
    key, idx = rng.choice(flat)
    x1, y1, x2, y2 = bbs[key].pop(idx)
    w, h = x2 - x1, y2 - y1
    pieces = []
    for _ in range(rng.randint(2, 3)):
        sw, sh = rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h
        sx = rng.uniform(x1, max(x1, x2 - sw))
        sy = rng.uniform(y1, max(y1, y2 - sh))
        box = _clip_box([sx, sy, sx + sw, sy + sh], img_w, img_h)
        if box[2] > box[0] and box[3] > box[1]:
            pieces.append(box)
    if not pieces:
        return None
    bbs[key].extend(pieces)
    return bbs


def op_merge(bbs, rng, img_w, img_h):
    """Merge two same-class boxes into their union (error type 6)."""
    candidates = [k for k in range(NUM_CLASSES) if len(bbs[k]) >= 2]
    if not candidates:
        return None
    key = rng.choice(candidates)
    i, j = rng.sample(range(len(bbs[key])), 2)
    a, b = bbs[key][i], bbs[key][j]
    union = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
    for k in sorted((i, j), reverse=True):
        del bbs[key][k]
    bbs[key].append(_clip_box(union, img_w, img_h))
    return bbs


# COORD_OPS keep bbs[k][i] aligned with regions[k][i] and run first.
# STRUCT_OPS move boxes between classes or change their number, which ends
# that alignment, so they run afterwards.
COORD_OPS = (op_perturb, op_translate, op_scale)
STRUCT_OPS = (op_swap_class, op_erase, op_add, op_split, op_merge)


def _mix_config(mix):
    if mix is None:
        return SEVERITY_MIX['balanced']
    return SEVERITY_MIX[mix] if isinstance(mix, str) else mix


def _choose_severity(weights, rng):
    r, acc = rng.random(), 0.0
    for name, weight in weights.items():
        acc += weight
        if r <= acc:
            return name
    return 'gross'


def corrupt(regions, rng, img_w, img_h, mix=None):
    """Bad candidate with a controlled severity.  Returns (bbs, severity)."""
    cfg = _mix_config(mix)
    for _ in range(30):
        severity = _choose_severity(cfg['weights'], rng)
        bbs = good_label(regions, rng)
        structural = rng.random() < cfg['p_struct']
        out = bbs if structural else rng.choice(COORD_OPS)(
            bbs, regions, rng, severity, img_w, img_h)
        n_struct = int(structural)
        if rng.random() < cfg['p_compose']:
            n_struct += rng.randint(1, 3)
        for _ in range(n_struct):
            if out is None:
                break
            out = rng.choice(STRUCT_OPS)(out, rng, img_w, img_h) or out
        if out is None or _total(out) == 0 or not _valid(out):
            continue
        if is_label_negative(out, regions):
            return out, ('count' if structural else severity)
    # guaranteed-negative fallback: relabelling a box always breaks the
    # per-class counts, and never produces an empty raster.
    bbs = good_label(regions, rng)
    out = op_swap_class(bbs, rng, img_w, img_h)
    return (out, 'count') if out is not None else (bbs, 'gross')


# ----------------------------------------------------------- augmentation --
def build_augment():
    """Geometry- and photometry-preserving augmentation.

    Limited to transforms that keep an image on the ImageNet manifold the
    frozen trunk expects, and that leave the image-label relationship intact:
    no channel shuffling, no 90-degree rotation, and no cut-out, since erasing
    an object while keeping its box would invert the supervision.
    """
    import albumentations as A
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(-0.06, 0.06),
                 rotate=(-10, 10), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2,
                                   p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=10, p=0.3),
        A.OneOf([A.GaussianBlur(), A.MotionBlur()], p=0.2),
        A.ImageCompression(quality_range=(50, 95), p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'],
                               min_visibility=0.3, clip=True,
                               filter_invalid_bboxes=True))




# ---------------------------------------------------- image-hard sources --
# Context built once over the training pool: a per-class bank of real
# ground-truth boxes (normalised coords) and a class co-occurrence matrix.
_BANK = None
_COOC = None

IMAGEHARD_SHARE = 0.5     # of non-detector negatives -> .45/.45/.10 of the
                          # non-swap negatives (swap_share is drawn first)


def build_context(pool):
    """pool: iterable of (image_path, packed {cls: [[x, y, w, h], ...]},
    img_w, img_h). Coordinates are original-image pixels."""
    global _BANK, _COOC
    _BANK = [[] for _ in range(NUM_CLASSES)]
    _COOC = np.zeros((NUM_CLASSES, NUM_CLASSES), np.float64)
    for _, packed, img_w, img_h in pool:
        present = [int(c) for c, b in packed.items()
                   if int(c) < NUM_CLASSES and b]
        for c in present:
            for x, y, w, h in packed[c]:
                if w > 0 and h > 0:
                    _BANK[c].append((x / img_w, y / img_h,
                                     w / img_w, h / img_h))
            for d in present:
                if d != c:
                    _COOC[c][d] += 1
    row = _COOC.sum(1, keepdims=True)
    _COOC = _COOC / np.maximum(row, 1.0)
    return _BANK, _COOC


def _plausible_absent(present, rng):
    """An absent class weighted by co-occurrence with the present ones."""
    absent = [k for k in range(NUM_CLASSES) if k not in present]
    if not absent:
        return rng.choice(sorted(present))
    if _COOC is None:
        return rng.choice(absent)
    scores = [sum(_COOC[c][k] for c in present) for k in absent]
    total = sum(scores)
    if total <= 0:
        return rng.choice(absent)
    r, acc = rng.random() * total, 0.0
    for k, sc in zip(absent, scores):
        acc += sc
        if r <= acc:
            return k
    return absent[-1]


def _bank_box(cls, rng, img_w, img_h):
    if not _BANK or not _BANK[cls]:
        return None
    x, y, w, h = _BANK[cls][rng.randrange(len(_BANK[cls]))]
    box = _clip_box([x * img_w, y * img_h,
                     (x + w) * img_w, (y + h) * img_h], img_w, img_h)
    return box if box[2] > box[0] and box[3] > box[1] else None


def ih_delete_one(bbs, regions, rng, img_w, img_h):
    """Exactly one box removed -- the subtlest count error."""
    flat = _flat(bbs)
    if len(flat) < 2:
        return None
    key, idx = rng.choice(flat)
    del bbs[key][idx]
    return bbs


def ih_swap_plausible(bbs, regions, rng, img_w, img_h):
    """One box relabelled to a co-occurrence-plausible absent class --
    undetectable from class statistics alone."""
    flat = _flat(bbs)
    if not flat:
        return None
    present = {k for k, _ in flat}
    key, idx = rng.choice(flat)
    new_key = _plausible_absent(present, rng)
    if new_key == key:
        return None
    bbs[new_key].append(bbs[key].pop(idx))
    return bbs


def ih_relocate(bbs, regions, rng, img_w, img_h):
    """One box moved to a random position, size and class kept -- invisible
    to per-class counts."""
    flat = _flat(bbs)
    if not flat:
        return None
    key, idx = rng.choice(flat)
    x1, y1, x2, y2 = bbs[key][idx]
    w, h = min(x2 - x1, img_w - 1), min(y2 - y1, img_h - 1)
    for _ in range(20):
        nx = rng.uniform(0, img_w - w)
        ny = rng.uniform(0, img_h - h)
        box = _clip_box([nx, ny, nx + w, ny + h], img_w, img_h)
        if box[2] > box[0] and box[3] > box[1]:
            bbs[key][idx] = box
            return bbs
    return None


def ih_add_bank(bbs, regions, rng, img_w, img_h, plausible_absent=False):
    """1-3 real boxes from the bank: present classes (B1) or plausible
    absent ones (B2) -- real box statistics, so shape alone cannot give
    the error away."""
    flat = _flat(bbs)
    if not flat:
        return None
    present = {k for k, _ in flat}
    for _ in range(rng.randint(1, 3)):
        cls = (_plausible_absent(present, rng) if plausible_absent
               else rng.choice(sorted(present)))
        box = _bank_box(cls, rng, img_w, img_h)
        if box is None:
            w = rng.uniform(0.08, 0.5) * img_w
            h = rng.uniform(0.08, 0.5) * img_h
            x1 = rng.uniform(0, max(1.0, img_w - w))
            box = _clip_box([x1, rng.uniform(0, max(1.0, img_h - h)),
                             x1 + w, rng.uniform(0, max(1.0, img_h - h)) + h],
                            img_w, img_h)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
        bbs[cls].append(box)
    return bbs


IH_OPS = (ih_delete_one, ih_swap_plausible, ih_relocate,
          lambda b, r, g, w, h: ih_add_bank(b, r, g, w, h, False),
          lambda b, r, g, w, h: ih_add_bank(b, r, g, w, h, True))
SEVERITY_CODES['imagehard'] = 7
CODE_NAMES[7] = 'imagehard'
SEVERITY_CODES['composed'] = 8
CODE_NAMES[8] = 'composed'
SEVERITY_CODES['imageswap'] = 9
CODE_NAMES[9] = 'imageswap'

# Probability that a negative gets 1-3 additional distinct error ops
# stacked on top of its primary error: the test protocol's Table 4
# draws candidates carrying 2-5 error types at once.
COMPOSE_PROB = 0.5

# Ops that can stack after alignment with the regions is already broken:
# the image-hard set plus the structural severity ops, signatures
# normalised to (bbs, regions, rng, img_w, img_h).
COMPOSE_OPS = tuple(IH_OPS) + tuple(
    (lambda op: lambda b, r, g, w, h: op(b, g, w, h))(op)
    for op in STRUCT_OPS)


def _stack_errors(bbs, regions, rng, img_w, img_h):
    """1-3 extra ops on a copy; None if the result stops being a
    metric-verified negative."""
    out = {k: [list(b) for b in v] for k, v in bbs.items()}
    for _ in range(rng.randint(1, 3)):
        out = rng.choice(COMPOSE_OPS)(out, regions, rng, img_w, img_h) or out
    if _total(out) == 0 or not _valid(out):
        return None
    return out if is_label_negative(out, regions) else None


def corrupt_unified(regions, rng, img_w, img_h, mix=None,
                    ih_share=IMAGEHARD_SHARE, compose=COMPOSE_PROB):
    """The unified negative mixer: image-hard with probability ih_share,
    the severity generator otherwise; with probability `compose` the
    negative carries 1-3 additional stacked errors. Always
    metric-verified; stacking falls back to the primary negative if it
    would stop verifying."""
    if rng.random() >= ih_share:
        bbs, tag = corrupt(regions, rng, img_w, img_h, mix)
    else:
        bbs, tag = None, 'imagehard'
        for _ in range(30):
            cand = good_label(regions, rng)
            out = rng.choice(IH_OPS)(cand, regions, rng, img_w, img_h)
            if out is None or _total(out) == 0 or not _valid(out):
                continue
            if is_label_negative(out, regions):
                bbs = out
                break
        if bbs is None:
            bbs, tag = corrupt(regions, rng, img_w, img_h, mix)
    if compose > 0 and rng.random() < compose:
        stacked = _stack_errors(bbs, regions, rng, img_w, img_h)
        if stacked is not None:
            return stacked, 'composed'
    return bbs, tag


# -------------------------------------------------------------- dataset ----
class Data(Dataset):
    """Balanced good/bad candidates over annotation files.

    paths is a list of (image_path, packed, requested_label); packed is
    {class_index: [[x, y, w, h], ...]} in original-image pixels; 1 asks
    for a good draw, 0 for a bad one. The returned label is always
    re-derived with the uncertainty-region metric, so a detector candidate
    that happens to satisfy it counts as good.

    Candidate draws are deterministic in (seed, epoch, index) -- the
    augmentation, when enabled, is not seeded: call set_epoch each
    epoch to regenerate candidates dynamically; leave it at 0 for a fixed
    set (validation). gt_map_grid > 0 additionally returns the
    ground-truth class map at that grid for the auxiliary supervision.
    """

    def __init__(self, paths, augment=False, detector_share=DETECTOR_SHARE,
                 deterministic=False, seed=0, severity_mix='balanced',
                 res=RES, return_meta=False, aug_prob=0.5,
                 ih_share=IMAGEHARD_SHARE, compose=COMPOSE_PROB,
                 gt_map_grid=0, sparse_planes=False,
                 swap_share=0.0, per_box=False, gt_counts=False,
                 cache_images=False):
        self.paths = paths
        self.ih_share = ih_share
        self.compose = compose
        self.sparse_planes = sparse_planes
        self.gt_map_grid = gt_map_grid
        # swap_share: fraction of negatives that pair a metric-good label
        # with a DIFFERENT pool image -- the grounding negative; 0 keeps
        # the plain mixer and consumes no rng (run_qa.sh uses 0.1).
        # per_box appends the
        # (N, BOX_FEATS) verification rows as the sample's last element.
        self.swap_share = swap_share
        self.per_box = per_box
        # cache_images keeps each worker's decoded+resized images in RAM
        # across epochs (needs persistent workers to pay off; per worker
        # ~0.9 MB per image for COCO at 640 px, ~0.1 MB for VOC at 224 px). Off by default; serving from the cache
        # is read-only for every downstream path.
        self.cache_images = cache_images
        self._img_cache = {}
        # gt_counts appends per-class GROUND-TRUTH box counts plus a
        # validity flag (0 under an image swap, whose label-side counts
        # do not describe the shipped image); consumes no rng.
        self.gt_counts = gt_counts
        self.epoch = 0
        self.augment = augment
        self.aug_prob = aug_prob
        self.detector_share = detector_share
        self.deterministic = deterministic
        self.seed = seed
        self.mix = _mix_config(severity_mix)
        self.res = res
        self.return_meta = return_meta
        self.pred_cache = {}
        self._models = None
        self._transform = build_augment() if augment else None

    def __len__(self):
        return len(self.paths)

    def set_epoch(self, epoch):
        # a shared value so persistent DataLoader workers see the new
        # epoch: worker processes hold copies of this object, and a
        # plain attribute write in the parent would never reach them
        self._epoch_shared.value = epoch

    @property
    def epoch(self):
        return self._epoch_shared.value

    @epoch.setter
    def epoch(self, value):
        import multiprocessing as _mp
        if "_epoch_shared" not in self.__dict__:
            self.__dict__["_epoch_shared"] = _mp.Value("i", 0)
        self._epoch_shared.value = value

    def _read_image(self, img_path):
        """(resized image, original h, original w), cached per worker
        when cache_images is on. Callers treat the array as read-only."""
        if self.cache_images and img_path in self._img_cache:
            return self._img_cache[img_path]
        raw = cv2.imread(img_path)
        if raw is None:
            raise FileNotFoundError(
                f"No such file or unreadable image: {img_path}")
        entry = (resize_keep_aspect(raw, self.res),) + raw.shape[:2]
        if self.cache_images:
            self._img_cache[img_path] = entry
        return entry

    def _rng(self, index):
        if self.deterministic:
            return random.Random(self.seed * 1_000_003
                                 + self.epoch * 7_776_146_593 + index)
        return random

    # -- detector candidates -------------------------------------------------
    def _detectors(self):
        if self._models is None:
            from torchvision.models.detection import (
                fasterrcnn_mobilenet_v3_large_320_fpn,
                FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                ssdlite320_mobilenet_v3_large,
                SSDLite320_MobileNet_V3_Large_Weights)
            w1 = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            w2 = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
            self._models = [
                (ssdlite320_mobilenet_v3_large(weights=w1).eval(), w1),
                (fasterrcnn_mobilenet_v3_large_320_fpn(weights=w2).eval(), w2)]
        return self._models

    def _detector_bbs(self, img_path, old_w, old_h, new_w, new_h, rng):
        """Detector boxes in *resized* coordinates, before any augmentation."""
        import torch
        from torchvision.io.image import read_image
        model, weights = rng.choice(self._detectors())
        cache_key = (model.__class__.__name__, img_path)
        if cache_key not in self.pred_cache:
            with torch.no_grad():
                out = model([weights.transforms()(read_image(img_path))])[0]
            self.pred_cache[cache_key] = {
                'scores': out['scores'].numpy(), 'boxes': out['boxes'].numpy(),
                'labels': out['labels'].numpy().astype(np.int32)}
        pred = self.pred_cache[cache_key]
        names = [weights.meta["categories"][i] for i in pred['labels']]
        bbs = {key: [] for key in range(NUM_CLASSES)}
        for idx, box in enumerate(pred['boxes']):
            name = DETECTOR_NAME_MAP.get(names[idx], names[idx])
            if name not in MAPPING or float(pred['scores'][idx]) <= \
                    DETECTOR_CONFIDENCE:
                continue
            b = _clip_box([box[0] / old_w * new_w, box[1] / old_h * new_h,
                           box[2] / old_w * new_w, box[3] / old_h * new_h],
                          new_w, new_h)
            if b[2] > b[0] and b[3] > b[1]:
                bbs[MAPPING[name]].append(b)
        return bbs

    # -- sample --------------------------------------------------------------
    def __getitem__(self, index):
        img_path, packed, requested = self.paths[index]
        rng = self._rng(index)

        image, old_h, old_w = self._read_image(img_path)
        new_h, new_w = image.shape[:2]

        boxes = boxes_from_packed(packed, old_w, old_h, new_w, new_h)

        # grounding negatives replace the image, not the label, so they
        # take neither the detector nor the corruption path
        want_swap = (requested == 0 and self.swap_share > 0
                     and rng.random() < self.swap_share)

        # the candidate source is chosen before augmenting, so detector boxes
        # pass through the same geometric transform as the ground truth
        use_detector = (not want_swap and requested == 0
                        and self.detector_share > 0
                        and rng.random() < self.detector_share)
        detector_bbs = None
        if use_detector:
            detector_bbs = self._detector_bbs(img_path, old_w, old_h,
                                              new_w, new_h, rng)

        if self.augment and rng.random() < self.aug_prob:
            image, boxes, detector_bbs = self._augment(image, boxes,
                                                       detector_bbs)
            new_h, new_w = image.shape[:2]

        if not any(len(v) for v in boxes.values()):
            # every ground-truth box fell outside the augmented frame; keep
            # the unaugmented sample so the raster is never empty
            image = self._read_image(img_path)[0]
            new_h, new_w = image.shape[:2]
            boxes = boxes_from_packed(packed, old_w, old_h, new_w, new_h)
            detector_bbs = None

        regions = build_regions(boxes, new_w, new_h)

        swap_img = self._swap_image(index, rng, new_w, new_h) \
            if want_swap else None
        if requested == 1:
            bbs, tag = good_label(regions, rng), 'good'
        elif swap_img is not None:
            # metric-good label, wrong image: bad as a pair, and the only
            # negative the label geometry alone can never reveal
            bbs, tag = good_label(regions, rng), 'imageswap'
            image = swap_img
        elif detector_bbs is not None and _total(detector_bbs) > 0:
            bbs, tag = detector_bbs, 'detector'
        else:
            bbs, tag = corrupt_unified(regions, rng, new_w, new_h,
                                       self.mix, self.ih_share,
                                       self.compose)

        label = 0 if is_label_negative(bbs, regions) else 1
        if tag == 'imageswap':
            label = 0
        if self.sparse_planes:
            # Ship only the classes that hold boxes -- the dense
            # 80-channel raster is ~99% zeros and dominates worker time
            # and shared memory; to_dense_batch scatters on the GPU.
            present = sorted(c for c, v in bbs.items() if v)
            sparse = np.zeros((len(present), self.res, self.res), np.uint8)
            for j, c in enumerate(present):
                sparse[j] = (render_plane(bbs[c], new_h, new_w, self.res)
                             * PLANE_SCALE).astype(np.uint8)
            planes = (np.array(present, np.int64), sparse)
            img = center_pad_uint8(image, self.res)
        else:
            planes = (render_planes(bbs, new_h, new_w, self.res)
                      * PLANE_SCALE).astype(np.uint8)
            img = normalize_image(image, self.res).astype(np.float32)
        extras = []
        if self.gt_map_grid:
            # Ground-truth class map at the fusion grid for the auxiliary
            # supervision -- built from the (augmented) GROUND TRUTH, never
            # the candidate. Boxes live in the resized (new_h, new_w) frame,
            # which center_pad places at an offset inside res x res.
            g = self.gt_map_grid
            oy, ox = (self.res - new_h) // 2, (self.res - new_w) // 2
            gt_map = np.full((g, g), 255, np.uint8)
            f = g / float(self.res)
            for cls in range(NUM_CLASSES):
                for x1, y1, x2, y2 in boxes[cls]:
                    gx1, gy1 = int((x1 + ox) * f), int((y1 + oy) * f)
                    gx2, gy2 = int((x2 + ox) * f), int((y2 + oy) * f)
                    gt_map[gy1:max(gy1 + 1, gy2 + 1),
                           gx1:max(gx1 + 1, gx2 + 1)] = cls
            extras.append(gt_map)
        if self.return_meta:
            extras.append(SEVERITY_CODES[tag])
        if self.gt_counts:
            # Per-class counts of the (augmented) GROUND-TRUTH boxes --
            # the count head learns what the image contains, never the
            # candidate. Last element is the validity flag.
            cnt = np.zeros(NUM_CLASSES + 1, np.float32)
            for cls in range(NUM_CLASSES):
                cnt[cls] = len(boxes[cls])
            cnt[-1] = 0.0 if tag == 'imageswap' else 1.0
            extras.append(cnt)
        if self.per_box:
            # letterboxed RoI rows + targets, aligned with _flat(bbs);
            # under a swapped image no claim is supported and the
            # geometric margins are void
            oy, ox = (self.res - new_h) // 2, (self.res - new_w) // 2
            ok, marg, mval = per_box_targets(bbs, regions)
            arr = np.zeros((len(ok), BOX_FEATS), np.float32)
            for row, (key, bi) in zip(arr, _flat(bbs)):
                x1, y1, x2, y2 = bbs[key][bi]
                row[:5] = (x1 + ox, y1 + oy, x2 + ox, y2 + oy, key)
            arr[:, 5] = ok
            arr[:, 6] = marg
            arr[:, 7] = mval
            if tag == 'imageswap':
                arr[:, 5] = 0.0
                arr[:, 7] = 0.0
            extras.append(arr)
        return (img, planes, label, *extras)

    def _swap_image(self, index, rng, new_w, new_h):
        """A different pool image resized to this sample's frame; None if
        no distinct readable image is found (the caller then falls back
        to an ordinary corruption negative)."""
        own = self.paths[index][0]
        for _ in range(10):
            other = self.paths[rng.randrange(len(self.paths))][0]
            if other != own:
                img = cv2.imread(other)
                if img is not None:
                    return cv2.resize(img, (new_w, new_h),
                                      interpolation=cv2.INTER_AREA)
        return None

    def _augment(self, image, boxes, detector_bbs):
        """Transform image, ground truth and detector boxes together."""
        flat, tags = [], []
        for key, lst in boxes.items():
            for b in lst:
                flat.append(b)
                tags.append(('gt', key))
        if detector_bbs is not None:
            for key, lst in detector_bbs.items():
                for b in lst:
                    flat.append(b)
                    tags.append(('det', key))
        if not flat:
            return image, boxes, detector_bbs
        try:
            out = self._transform(image=image, bboxes=flat,
                                  labels=list(range(len(flat))))
        except Exception:
            return image, boxes, detector_bbs
        new_boxes = {key: [] for key in range(NUM_CLASSES)}
        new_det = None if detector_bbs is None else \
            {key: [] for key in range(NUM_CLASSES)}
        for box, tag_idx in zip(out['bboxes'], out['labels']):
            kind, key = tags[int(tag_idx)]
            box = [float(c) for c in box]
            if box[2] - box[0] < MIN_BOX_PX or box[3] - box[1] < MIN_BOX_PX:
                continue
            (new_boxes if kind == 'gt' else new_det)[key].append(box)
        return out['image'], new_boxes, new_det


def get_sample(image, annotation, res=RES):
    """Render one (BGR image, {class_name: [[x1,y1,x2,y2], ...]}) pair.

    Returns (img, planes), both CHW float32, matching the training transform.
    """
    old_h, old_w = image.shape[:2]
    image = resize_keep_aspect(image, res)
    new_h, new_w = image.shape[:2]
    bbs = {}
    for name, boxes in annotation.items():
        if name not in MAPPING:
            raise KeyError(f"unknown class {name!r}; "
                           f"expected one of {sorted(MAPPING)}")
        for x1, y1, x2, y2 in boxes:
            bbs.setdefault(MAPPING[name], []).append([
                int(np.clip(min(x1, x2) / old_w * new_w, 0, new_w - 1)),
                int(np.clip(min(y1, y2) / old_h * new_h, 0, new_h - 1)),
                int(np.clip(max(x1, x2) / old_w * new_w, 0, new_w - 1)),
                int(np.clip(max(y1, y2) / old_h * new_h, 0, new_h - 1))])
    return normalize_image(image, res), render_planes(bbs, new_h, new_w, res)


def box_rois(annotation, old_w, old_h, res=RES):
    """RoI rows for the per-box head from a {class_name: [[x1,y1,x2,y2]]}
    annotation, letterboxed exactly as get_sample renders: (N, 5)
    float32 rows [x1, y1, x2, y2, class_index]. Callers prepend the
    sample index column the model's `boxes` input expects.
    """
    s = res / max(old_w, old_h)
    new_w, new_h = max(1, int(old_w * s)), max(1, int(old_h * s))
    oy, ox = (res - new_h) // 2, (res - new_w) // 2
    rows = []
    for name, boxes in annotation.items():
        if name not in MAPPING:
            raise KeyError(f"unknown class {name!r}; "
                           f"expected one of {sorted(MAPPING)}")
        for x1, y1, x2, y2 in boxes:
            rows.append([
                int(np.clip(min(x1, x2) / old_w * new_w, 0, new_w - 1)) + ox,
                int(np.clip(min(y1, y2) / old_h * new_h, 0, new_h - 1)) + oy,
                int(np.clip(max(x1, x2) / old_w * new_w, 0, new_w - 1)) + ox,
                int(np.clip(max(y1, y2) / old_h * new_h, 0, new_h - 1)) + oy,
                MAPPING[name]])
    return np.array(rows, np.float32).reshape(-1, 5)
