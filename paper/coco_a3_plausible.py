"""A3 swaps to a context-plausible class instead of a uniform random one
(README item 6: the image-hard generator's A3 and A2 come from here).

Table 1 fixes only that A3 "swap classes of labels from uncertainty
regions"; the replacement distribution is unspecified. Drawing uniformly
from the other 79 classes almost always injects a class that never
co-occurs with the scene, so the negative is detectable from the label
alone -- a zebra in a kitchen. A
replacement drawn from classes that plausibly co-occur with what is
already labelled removes that route: the class set stays ordinary, so the
only way to catch the error is to check the image and find no such object.

The target is drawn from classes ABSENT from the ground truth, which also
guarantees the result is genuinely negative -- no object of that class
exists anywhere, so the box cannot accidentally satisfy some other
region's Equations 1-4.

Import-only module: importing it patches data_loader.corrupt; call
build_cooccurrence() before use (train_coco_corr.py does both).
"""

import data_loader as dl

SMOOTH = 0.5

# Floor of A2's offset draw, as a fraction of the box dimension. The paper's
# 0.02 puts a negative 2% of the box away from the nearest legal positive --
# 0.74px for a 37px box -- so for small objects the two classes are separated
# by less than a pixel and A2 is at chance no matter what the model does.
# Raising the floor gives the classes a real margin without changing what
# either class means: good is still "inside the uncertainty region", bad is
# still "outside it".
A2_MIN_OFFSET = 0.02

_cooc = None      # cooc[a][b] = images containing both a and b
_freq = None      # freq[a]    = images containing a
_total = 1        # images in the pool


def build_cooccurrence(annotations_path):
    """Class co-occurrence counted over the pool's ground truth."""
    global _cooc, _freq, _total
    samples, _ = dl.load_coco(annotations_path)
    _total = len(samples)
    n = dl.NUM_CLASSES
    _cooc = [[0] * n for _ in range(n)]
    _freq = [0] * n
    for _, ann in samples:
        present = sorted(ann)
        for a in present:
            _freq[a] += 1
            for b in present:
                if a != b:
                    _cooc[a][b] += 1
    return _cooc, _freq


def plausible_classes(annotation, context=None):
    """Score every absent class by how ordinary it would look beside the
    classes already labelled.

    The score is the WEAKEST pairwise association with any present class,
    because that is the statistic a label-only detector keys on: one
    implausible pair (a zebra beside an oven) gives the whole label away,
    however plausible the rest. Scoring by an average instead leaves that
    minimum free to be tiny.
    """
    # plausibility is judged against the label as it stands, so that a
    # second swap cannot add a class that only looks reasonable beside the
    # original scene; admissibility must never be spent twice
    present = sorted(annotation if context is None else context)
    absent = [c for c in range(dl.NUM_CLASSES) if c not in annotation]

    def pmi(a, b):
        # Jaccard, not PMI: pointwise mutual information divides by the
        # candidate's own frequency, so a class appearing in a handful of
        # images scores high on smoothing alone.
        union = _freq[a] + _freq[b] - _cooc[a][b]
        return _cooc[a][b] / union if union else 0.0

    # how ordinary the ground-truth class set already looks; a replacement
    # is only admissible if it does not make the label look stranger, which
    # is what keeps negatives and positives on the same side of the cue
    base = min((pmi(a, b) for i, a in enumerate(present)
                for b in present[i + 1:]), default=None)
    scores = []
    for c in absent:
        worst = min((pmi(c, p) for p in present), default=1.0)
        scores.append(worst)
    if base is not None:
        keep = [(c, s) for c, s in zip(absent, scores) if s >= base]
        if keep:
            return [c for c, _ in keep], [s for _, s in keep]
    # no admissible class: fall back to the most plausible few
    ranked = sorted(zip(absent, scores), key=lambda t: -t[1])[:8]
    return [c for c, _ in ranked], [s for _, s in ranked]


def corrupt_a3_plausible(annotation, rng, img_w, img_h, subtype=None):
    """data_loader.corrupt with only A3's target distribution changed."""
    out = dl.good(annotation, rng, img_w, img_h)
    flat = dl._flat(out)
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
        r = dl._ranges(annotation[c][i], img_w, img_h)
        gdims = (gw, gh, gw, gh)
        idims = (img_w, img_h, img_w, img_h)
        for j in rng.sample(range(4), rng.randint(1, 4)):
            off = rng.uniform(A2_MIN_OFFSET, dl.REGION) * gdims[j]
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
        # the only change from data_loader.corrupt: one replacement class
        # for the whole error, drawn from those that sit plausibly beside
        # what is already labelled. Relabelling every affected box to the
        # same class also keeps the label's class count close to the
        # scene's, so the error cannot be read off the class list.
        candidates, scores = plausible_classes(annotation)
        picks = sorted(rng.sample(flat, rng.randint(1, len(flat))),
                       key=lambda t: -t[1])
        if candidates:
            c2 = rng.choices(candidates, weights=scores, k=1)[0]
        else:
            c2 = rng.choice([k for k in range(dl.NUM_CLASSES)
                             if k != picks[0][0]])
        for c, i in picks:
            out.setdefault(c2, []).append(out[c].pop(i))
            if not out[c]:
                del out[c]
    else:
        present = sorted(annotation)
        absent = [k for k in range(dl.NUM_CLASSES) if k not in annotation]
        for _ in range(rng.randint(1, 3)):
            if subtype == "B1" or not absent:
                cls = rng.choice(present)
            else:
                cls = rng.choice(absent)
            for _ in range(20):
                box = dl._random_box(rng, img_w, img_h)
                if not dl._inside_any_region(box, annotation, img_w, img_h):
                    break
            out.setdefault(cls, []).append(box)
    return out, subtype


dl.corrupt = corrupt_a3_plausible
