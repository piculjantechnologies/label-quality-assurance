# Label quality assurance on COCO: code for the 2023 Applied Sciences paper

Code for the method of

> Pičuljan, N. and Car, Ž. *Machine Learning-Based Label Quality Assurance for
> Object Detection Projects in Requirements Engineering.* Applied Sciences
> 13(10):6234, 2023. [doi:10.3390/app13106234](https://doi.org/10.3390/app13106234)

with implementation details documented by the paper's first author:
two interpretations of the paper's text (cross-entropy on logits, the
box-scaled region; items 1–2) and ten additions to the method and its
training (exact-ground-truth positives, correlation fusion, an
ImageNet-pretrained image trunk, image-hard negatives, weight EMA, ghost
BatchNorm, the raster head, the held-out refit, image-swap negatives,
detector candidates; items 3–12), each listed with the reason for it
under
[Implementation details](#implementation-details-and-why-each-one-is-needed).
Everything not listed is the paper's method unchanged: the task (given
an image and a candidate label, judge good/bad), the model's two inputs
(the 640-px letterboxed image and the
label raster with one hollow-border plane per class), the paper's two
ResNet-18 trunks and fully connected layers, the 0.2 uncertainty region
(box-scaled, interpretation 2), the A1–A3/B1–B2 error taxonomy and
Section 5's error probabilities, the val2017-trains / train2017-tests
split discipline, the effective batch of 64, and the balanced good/bad
draw of the test protocol. The training settings that depart from
Section 5 are tabulated under
[The method, as released](#the-method-as-released).

On this release's implementation of the paper's Section 5 test protocol
(108 151 train2017 candidates), the released checkpoint reaches **0.8143
accuracy against the paper's reported 0.8166** — slightly below the
paper's, from a single run each — with every metric cell of the
paper's Table 2 within 0.01. [Results](#results) gives the tables and
what the comparison does and does not establish.

## The method, as released

**Goodness metric.** A candidate label is judged against the image's
ground truth by Equations 1–4: each coordinate of a ground-truth box may
move outward by up to 0.2 × that box's width or height, never inward,
and the region is clipped to the image (`data_loader._ranges`; the
box-scaled reading, interpretation 2). The error generators label their
candidates bad, as the paper's generator does: each error takes a box out
of its region, deletes boxes, relabels them or adds boxes placed outside every
region (up to 20 random placements per box, the last kept if none finds
room). A good candidate keeps every box inside its own region.
Candidates that no generator produced — detector output (item 12) — are
judged by the whole-label form (`data_loader.label_good`): good only if,
class by class, the candidate's boxes pair one-to-one with the
ground-truth boxes, each inside its own box's region. The raster head's
targets use the per-box form (`data_loader.box_good`): a box is good if a
ground-truth box of its class has a region that contains it.

**Candidates** (`data_loader.py`, `coco_hard_negatives.py`,
`coco_a3_plausible.py`). Every epoch passes through each training image
twice — one good-requested and one bad-requested candidate, drawn anew
from (seed, epoch, index). One sample in ten is first augmented with the
augmentation set of the author's thesis (Pičuljan, N. *Machine
learning-based method for quality assurance of object bounding box labels
in images.* PhD thesis, University of Zagreb, Faculty of Electrical
Engineering and Computing, 2025,
[urn:nbn:hr:168:865107](https://urn.nsk.hr/urn:nbn:hr:168:865107),
Listing 8.11; `data_loader.build_augment`), with the ground truth and any
detector boxes transformed together with the image; the unaugmented
sample is kept when no ground-truth box survives. Good candidates are
the exact ground truth half the time (item 3), else every box redrawn
uniformly inside its region. Negatives come from one mixer, drawn in this
order: 10% pair a good label of the image with another training image
(image-swap, item 11); 10% of the rest are detector output (item 12),
labelled by the whole-label metric, so a few of them are good; the
remainder splits evenly between the paper's Section 5 generator and the
image-hard generator (item 6) — about 10 / 9 / 40 / 40% of the
bad-requested candidates. Validation, the refit and the test draw from the paper's generator alone,
with good candidates inside the regions (never the verbatim ground truth)
and no augmentation.

**Architecture** (`coco_corrnet.py`). CorrNet runs the paper's two
ResNet-18 trunks to `layer4` (stride 32; 20 × 20 cells at 640 px). The
image trunk starts from ImageNet weights and receives the letterbox
normalised inside the model (item 5); the label trunk, trained from
scratch, reads the 80-plane raster. The classifier sees both trunks'
global-average-pooled maps (512 + 512), their element-wise product
averaged over the grid, and — on L2-normalised features — the product
averaged over the cells the label draws on and its per-channel maximum
(item 4): 5 × 512 = 2 560 features into the paper's fully connected
layers 512 → 256 → 128 → 2 (BatchNorm, ReLU and dropout between them).
The raster head (item 9) scores the label cell by cell on the stride-8
grid (80 × 80 cells) and adds a smooth minimum, the mean and
log(1 + count) of the claimed cells' support to the good logit.
Parameters: 26 786 535 — image trunk 11 176 512, label trunk
11 417 984, fully connected layers 1 477 506, raster head 2 714 533 —
against the 23 285 570 of the paper's "large" network (Table 3).

**Training and refit** (`train_coco_corr.py`, `refit.py`, `run.sh` — the
recipe's only carrier; the trainer's bare defaults are not the recipe).
AdamW with Section 5's learning rate, betas and eps (1 × 10⁻³,
(0.9, 0.999), 1 × 10⁻⁸) and weight decay 1 × 10⁻²; the image trunk at
0.1 × the learning rate (item 5); 2 linear warmup epochs, then cosine
decay to 0 at the last step, stepped per optimizer step; the effective
batch of 64 as 32 × 2 gradient accumulation with BatchNorm statistics
per group of 8 samples (item 8); mixed precision; a per-step weight EMA
of 0.998, whose weights are validated and saved (item 7). The loss is the
verdict's two-class cross-entropy plus 0.5 × each of the raster head's
two per-cell losses (item 9). All 160 epochs run, with no early
stopping, and `best_model.pth` is the epoch with the highest validation
ROC-AUC. Validation holds out 10% of val2017 (454 of 4 541 images; 4 087
train) and scores a fixed set drawn as the test is drawn — both
candidates of every held-out image over draw rounds 0–3, 3 632
candidates, the same every epoch. `refit.py` then refits the verdict
layer on the same 454 images over draw rounds 4–11 (item 10) and writes
the released checkpoint, `best_model_refit.pth`. The training settings
depart from Section 5 as follows; none of them changes what counts as a
good or bad label:

| | Section 5 | `run.sh` |
| --- | --- | --- |
| optimizer | Adam, betas (0.9, 0.999), eps 1 × 10⁻⁸, weight decay 0 | AdamW, the same betas and eps, weight decay 1 × 10⁻² |
| learning rate | 1 × 10⁻³ | 1 × 10⁻³ after 2 linear warmup epochs, cosine decay to 0; image trunk 1 × 10⁻⁴ (item 5) |
| batch | one batch of 64 | 64, as 32 × 2 gradient accumulation, BatchNorm statistics per 8 samples (item 8) |
| epochs | 250 | 160, no early stopping; the epoch with the highest validation ROC-AUC is kept |
| candidates per epoch | one per image, good or bad with equal probability | one good-requested and one bad-requested per image |
| augmentation | not stated | the thesis's Listing 8.11 set on one sample in ten (`--augment_p 0.1`) |
| precision | not stated | mixed precision (`--amp`) |
| weight averaging | not stated | per-step weight EMA 0.998 (item 7) |
| validation and selection | part of val2017, size not stated; validation curves in Figure 17 | 10% of val2017 (454 images), a fixed test-protocol draw of 3 632 candidates; selection on ROC-AUC |

**Speed.** `run.sh` trains the effective batch of 64 as 32 × 2
accumulation, with `GhostBatchNorm` supplying the statistics of groups
of 8 (about 23 GB of GPU memory, observed); `FAITHFUL=1` runs it as 8 × 8
accumulation for smaller GPUs, whose physical batches of 8 give the same
statistics. The two forms are not bit-identical: the per-cell losses are
averaged over each physical batch (32 or 8 samples), and `drop_last`
trims a different remainder of each epoch. Sparse plane transport
(`--sparse_planes`) ships only the class planes that hold boxes from the
loader workers — the dense 80 × 640 × 640 float32 raster is about 131 MB
per sample — and rebuilds the dense raster on the GPU; the model receives
bit-identical batches either way (`data_loader.collate_sparse_planes`,
`planes_to_dense`). The candidate draws — good/bad request, image-swap,
detector and generator choices — are a function of (seed, epoch, index);
the augmentation's own random draws are not seeded, so a re-run
reproduces the recipe in distribution, not bit for bit. On one H100
PCIe 80 GB the 160 epochs take about 4 h 25 min (as observed).

## Implementation details, and why each one is needed

**Interpretations (where the paper's text leaves room or disagrees with
its figures, the reading implemented here):**

1. **Cross-entropy on logits** (`neural_network.py`). The network graphs
   of Figures 18–19 end in a sigmoid over the two outputs, and Section 5
   trains with "the cross-entropy loss function". A sigmoid scores the two
   outputs independently rather than as a distribution over the two
   classes, so this code reads the objective as two-class cross-entropy:
   the model returns logits and the loss is `CrossEntropyLoss`, which
   applies the softmax itself.
2. **Box-scaled uncertainty region** (`data_loader._ranges`). Equations 1–4
   read "width × 0.2" and "height × 0.2", and the surrounding text glosses
   width and height as the input image's; Figures 9–11 draw the region at
   0.2 of the *box* extent (in Figure 9's 640 × 480 image, 71 px beside a
   359 px box, where the image reading gives 128 px), which is the
   intended definition. The region scales with the labelled box here;
   image bounds still clip it.

**Additions to the method and its training:**

3. **Positives include the exact ground truth** (`data_loader.good`,
   `--p_exact 0.5`: half the positives are the unmodified annotation, the
   other half are drawn inside the uncertainty regions as in the paper).
   Under a continuous re-draw inside the region, the unmodified
   annotation — the label a user actually submits — is a measure-zero
   event. Region-jittered positives can sit as close as 0.02 × box dim to
   an A2 negative, so crisp positives are what makes the boundary
   learnable. The base label that every generated negative corrupts is
   drawn the same way, so half of those negatives differ from the ground
   truth by their error alone.
4. **Correlation fusion** (`coco_corrnet.py`; `--corr_masked --corr_max
   --corr_cos`). The paper's pool-then-concat fusion pools each trunk
   before the branches meet, so no layer can compare a box with the
   object it annotates. CorrNet compares the two trunks' aligned
   stride-32 feature maps position by position — their product averaged
   over the grid, and on L2-normalised features averaged over the cells
   the label draws on and maximised per channel — so agreement between
   image content and claimed boxes, not label statistics alone, can carry
   the verdict. The first fully connected layer takes 2 560 inputs in
   place of the paper's 1 024.
5. **ImageNet-pretrained image trunk, normalised inside the model**
   (`--pretrained_image --input_norm --freeze_image_bn --img_lr_mult
   0.1`). Section 5 initialises both trunks randomly. Here the image
   trunk starts from torchvision's ImageNet ResNet-18 (`IMAGENET1K_V1`):
   the 4 087 training images are few for learning an image representation
   from nothing, and the correlation fusion (item 4) and the raster head
   (item 9) can only compare a label with image evidence the trunk
   provides. The pipeline feeds the paper's raw 0–255 BGR
   letterbox; the model flips it to RGB and applies the ImageNet mean and
   standard deviation inside `forward` (buffers `in_mean` and `in_std`,
   stored in the checkpoint; the letterbox's black padding maps to
   −mean/std, exactly as when a black-padded image is normalised), so the
   pretrained
   weights see the input statistics they were trained on and start
   exactly as published. The trunk's BatchNorm layers stay in eval mode
   during training, so their statistics remain the ImageNet ones rather
   than being re-estimated from groups of 8 letterboxed images (item 8),
   while every weight, the first convolution included, fine-tunes at
   0.1 × the learning rate (1 × 10⁻⁴; `image trunk lr 1.00e-04` in
   `train.out`), so the pretrained features are adapted rather than
   overwritten.
6. **Image-hard negatives, mixed with the paper's generator**
   (`--hard_negatives --paper_mix 0.5`, `coco_hard_negatives.py`;
   plausible class swaps via `coco_a3_plausible.py`). Of the negatives
   that are neither image-swaps (item 11) nor detector candidates
   (item 12), half come from the paper's own Section 5 generator
   (`data_loader.PAPER_CORRUPT`), half from `coco_hard_negatives`. The
   generator's purely synthetic corruptions leave the "looks plausible but
   wrong" region under-trained; the hard negatives are built from real
   boxes and co-occurrence-plausible classes. They keep the paper's 50/50
   type-A/type-B split but draw type A uniformly over four subtypes — A1
   deletes exactly one box, A2 is unchanged, A3 swaps to a
   co-occurrence-plausible class, and a fourth subtype R relocates one box
   off every ground-truth region (up to 20 placements; the last is kept if
   every one lands inside a region) — and their B1/B2 boxes are real
   ground-truth boxes of the class drawn from a per-class bank instead of
   the uniform random boxes this release's implementation of the paper's
   generator draws (the paper does not say how B boxes are placed). The
   co-occurrence tables and the box bank are built from the annotations
   of all 4 541 val2017 images, the 454 held out for validation and the
   refit included — their boxes and classes, never their pixels
   (`coco_a3_plausible.build_cooccurrence`,
   `coco_hard_negatives.build_box_bank`). Either generator alone teaches
   half the taxonomy: the bank
   boxes are what teach B detection, while the paper's generator keeps
   A1's multi-box deletions and its uniform A3 swaps, which the test
   draws.
7. **Weight EMA** (`--ema_decay 0.998`): a per-step exponential moving
   average of the weights, averaging about the last 500 optimizer steps
   (about four epochs of 128 steps); validation and the saved models
   (`best_model.pth`, `model.pth`) use the EMA weights, so every
   validation number reported here is an EMA-weights number. The average
   damps the step-to-step noise of the weights among which validation
   selects.
8. **Batch and BatchNorm** (`--ghost_bn 8`): `run.sh` trains the effective
   batch of 64 as 32 × 2 gradient accumulation with ghost BatchNorm, so
   BatchNorm statistics come from groups of 8 samples; `FAITHFUL=1` runs
   8 × 8 accumulation, whose physical batches of 8 give the same statistics.
   This departs from the paper, which trains one batch of 64 on one GPU,
   so with standard BatchNorm (Figure 18 shows BatchNorm layers; the text
   does not discuss it) its statistics span all 64 samples. The groups of
   8 apply to the label trunk and the fully connected layers; the image
   trunk keeps its ImageNet statistics (item 5), and the raster head
   normalises with GroupNorm, which does not depend on the batch.
9. **Raster head** (`coco_corrnet.py`; `--raster_head --box_weight 0.5
   --cls_weight 0.5`). A label is bad if any one of its boxes is wrong,
   but pooled features — the paper's fusion, and CorrNet's correlations,
   which are averaged over the grid — let one wrong box among many correct
   ones, and the empty rest of the image, average away. The head judges
   the label where it is drawn. On the stride-8 grid (80 × 80 cells for
   640 px) it reads the image trunk's stride-8 and stride-16 feature maps
   next to the label raster itself: which classes' outlines cross each
   cell (each class plane max-pooled 8 × 8) and where inside the cell they
   run (the class-agnostic outline, pixel-unshuffled into 64 channels, so
   a shift of one pixel is still visible). Four 3 × 3 convolution blocks
   with dilations 1, 2, 4 and 8 give each cell the context around a box
   edge; per cell they score the support for each of the 80 classes and,
   as a training signal, the class the image shows there (80 = nothing).
   The raster's own class planes select the claimed classes' support, and
   the verdict adds a smooth minimum and the mean of the claimed cells'
   support and log(1 + their count) to the good logit through a layer
   initialised at zero, so training starts from the plain CorrNet
   function. Two per-cell losses train the head next to the verdict's
   cross-entropy (`data_loader.cell_targets`): each claimed cell's support
   against the per-box form of Equations 1–4 for the box whose outline
   crosses it (the minimum where several cross), and every cell's class
   against the ground-truth class map. The model's inputs are unchanged —
   image and label raster, no box list — and at inference it returns only
   the two verdict logits. The head adds 2 714 533 parameters:
   26 786 535 in total, against the 23 285 570 of the paper's "large"
   network (Table 3).
10. **Held-out refit** (`refit.py`, the last step of `run.sh`). The
    verdict's margin is a weighted sum of the fully connected layers'
    margin and the head's three terms, plus a bias. Training fits these
    weights on images the model has partly memorised, and on a candidate
    mix (exact ground truth, image-hard, image-swap and detector
    candidates) that differs from the test protocol's; on unseen images
    drawn by the test protocol the verdict condemns too many good labels.
    `refit.py` refits the five numbers by logistic regression on held-out
    data — the 454 val2017 images the run never trains on (`--val_split
    0.1 --seed 0`, as `run.sh` passes them), with labels drawn by the
    paper's Section 5 generator and good candidates inside the regions
    (the test protocol), draw rounds 4–11 (validation uses rounds 0–3),
    7 264 candidates — and folds them into the last fully connected layer
    and the head's mixing layer, so the released checkpoint is an ordinary
    model with the same two inputs. The train2017 test pool is never used.
    Most of the effect is a shift of the decision threshold; the rest
    re-weights the head against the fully connected layers
    ([Results](#results)).
11. **Image-swap negatives** (`--swap_share 0.1`;
    `data_loader.LabelQualityDataset._other_image`). One negative in ten
    is a good label of the image — drawn as a positive is (item 3) —
    shown over another training image, stretched to this image's size,
    and labelled bad. Its geometry and class statistics are those of a
    positive, so only the image can reveal the error: it is the grounding
    negative that label geometry alone can never catch. The raster head's
    targets follow what the image shows: every claimed cell is bad, and
    the class map is the other image's ground truth
    (`data_loader.cell_targets`, `force_bad`, `class_annotation`).
12. **Detector candidates** (`--detector_share 0.1`;
    `data_loader.detector_boxes`, `data_loader.label_good`). Of the
    negatives that are not image-swaps, one in ten is a real detector's
    output instead of a generated error: one of two torchvision detectors
    with COCO weights — SSDLite320-MobileNetV3-Large or
    FasterRCNN-MobileNetV3-Large-320-FPN, drawn per candidate — keeping
    the boxes scored above 0.9, mapped to the pool's classes by name (the
    detectors run on the CPU inside the loader workers). Its errors are of
    the taxonomy's kinds — missed, loose, added and misclassified boxes —
    with the geometry and class confusions of a real model rather than of
    a generator. When the sample is augmented, the detector's boxes are
    transformed together with the ground truth. Since no generator chose
    an error, the candidate is labelled by the whole-label form of
    Equations 1–4 (`label_good`), which is how this code reads the
    equations' verdict on a whole label: good only if, class by class, the
    candidate has exactly as many boxes as the ground truth and they pair
    one-to-one with the ground-truth boxes, each candidate box inside its
    own ground-truth box's region (a bipartite matching found by
    augmenting paths). A missing box (A1), a box outside its region (A2),
    a wrong class (A3) and an added box (B1, B2) each break the pairing; a
    detection that matches the ground truth this closely is a good label,
    so a few candidates requested as bad are good ones. The raster head's
    per-cell targets use the per-box form (`data_loader.box_good`).

The trainer keeps research options outside the recipe — `--corr16`,
`--corr8`, `--corr4`, `--a2_min_offset`, `--a2_share`, `--warm`,
`--init`, `--hflip`, `--save_every`, `--limit`, the paper's Adam
(`--optimizer adam`, the trainer's default) and, for `--pretrained_image`
without `--input_norm`, a fold of the ImageNet input convention into the
first convolution, with that layer's learning rate scaled to match. The
shipped run uses none of them (`training_log.json` records every
argument).

## Layout

| file | role |
| --- | --- |
| `neural_network.py` | the paper's two-branch network, whose trunk layout CorrNet reuses (interpretation 1) |
| `data_loader.py` | COCO parsing, rendering, uncertainty regions, the paper's error generator, the per-box and whole-label metric, augmentation, image-swap and detector candidates, raster-head targets, sparse plane transport (interpretation 2; items 3, 9, 11, 12) |
| `coco_corrnet.py` | CorrNet: the correlation fusion over the paper's trunks, in-model input normalisation, the raster head, GhostBatchNorm (items 4, 5, 8, 9) |
| `coco_a3_plausible.py` | co-occurrence-plausible class swaps (item 6) |
| `coco_hard_negatives.py` | image-hard negatives and their mix with the paper's generator (item 6) |
| `train_coco_corr.py`, `run.sh` | the trainer and the recipe carrier, whose flags select items 3–9, 11 and 12; `run.sh` also runs the refit (item 10); interpretation 1, item 7 and item 5's learning-rate groups are implemented in the trainer |
| `refit.py` | the held-out refit of the verdict layer (item 10) |
| `evaluate.py` | scores a checkpoint on the paper's train2017 test protocol (Table 2 layout, Table 4-style subtype rows, ROC-AUC) |
| `demo_samples.py`, `samples/` | the five-sample demo (Good/Bad verdicts + expansion column); `samples/fetch_samples.py` downloads the images (~1 MB) |
| `interactive_demo.py` + `interactive_demo.html` | box editor with the model's live verdict, on COCO images or your own uploaded photos |
| `web_demo.py` | upload-and-score web UI mirroring the paper's published demo (needs `gradio>=6`; `--no-upload` = examples-only page and `--readme-url` = where its 'code release' link points, as the public instance runs) |
| `artifacts/` | the shipped run: `train.out`, `training_log.json`, `refit.out`, `refit.json`, `eval.out`, `eval.json`, `demo.out` (`best_model_refit.pth` is downloaded separately, see Weights) |
| `requirements.txt`, `LICENSE`, `NOTICE` | packaging |

## Setup

```bash
pip install -r requirements.txt

# COCO 2017 images and annotations from cocodataset.org;
# val2017 trains, train2017 is the held-out test pool
export COCO_IMAGES=/path/to/coco/val2017
export COCO_ANNOTATIONS=/path/to/coco/annotations/instances_val2017.json
```

COCO needs no preparation step: `COCO_IMAGES` names the raw image
directory and `COCO_ANNOTATIONS` the matching `instances_*.json`, and
`data_loader.load_coco` filters the pool when a script starts (see
[Data](#data)). `run.sh` and the box editor read the val2017 pair from
these variables; `evaluate.py` takes the train2017 pair as arguments.

## Run

```bash
./run.sh                # trains into artifacts/ (resumable), then refit.py
                        # writes artifacts/best_model_refit.pth
FAITHFUL=1 ./run.sh     # the same recipe as 8 x 8 accumulation

# the refit alone (the last step of run.sh), from a run's best_model.pth
python3 refit.py artifacts/best_model.pth \
    --images "$COCO_IMAGES" --annotations "$COCO_ANNOTATIONS" \
    --val_split 0.1 --seed 0 \
    --out artifacts/best_model_refit.pth --report artifacts/refit.json

# the paper's test protocol on train2017
python3 evaluate.py --checkpoint artifacts/best_model_refit.pth \
    --images /path/to/coco/train2017 \
    --annotations /path/to/coco/annotations/instances_train2017.json \
    --batch_size 32 --workers 22 --out artifacts/eval.json

python3 samples/fetch_samples.py            # one-time demo images
python3 demo_samples.py artifacts/best_model_refit.pth
python3 interactive_demo.py                 # box editor at 127.0.0.1:7864
                                            # (--checkpoint defaults to
                                            # artifacts/best_model_refit.pth)
python3 interactive_demo.py --uploads-only  # the same editor, own photos only
                                            # (no COCO needed)
pip install 'gradio>=6' && python3 web_demo.py
                                            # web UI, optional; add --no-upload
                                            # for an examples-only page
```

The box editor serves the run's held-out val2017 images — 454 for the
shipped run, whose seed and split it reads from the `training_log.json`
next to the checkpoint — and `--all` the whole val2017 pool.

In the box editor, "upload photo…" (or dropping an image on the page)
scores your own photo: draw and edit boxes with the 80 COCO classes, and
the model's P(good) follows every edit. A photo has no ground truth, so
the metric column reads n/a and only the model judges it. A photo larger
than COCO's scale is area-downscaled to a longest side of 640 px first;
the model then sees its 640-px letterbox, like every training image. The
page accepts files up to 8 MB. Uploaded photos are held only in server
memory for re-scoring — never written to disk or logged — and are
discarded 30 minutes after they were last scored, when 64 more recently
used photos are held, or when the server restarts.

**Deploying publicly.** The box editor is hardened for public exposure
but is not meant to face the internet by itself. Keep `--host 127.0.0.1`
and put a reverse proxy in front that terminates TLS, caps the request
body (`client_max_body_size 13m`), rate-limits per client (`limit_req`,
`limit_conn`) and bounds slow clients (`client_body_timeout`,
`proxy_read_timeout`). The server itself caps requests at 12 MB, checks
an image's dimensions from its header before decoding it, runs one
inference at a time, holds at most 32 connections with a 30 s socket
timeout, returns generic error text and writes a one-line access log.
Serve the demo at the root of its own host, or under a path prefix at a
trailing-slash URL — the page uses relative `api/` URLs.
`--permissive-licences` restricts the COCO pool to photographs under
CC BY, CC BY-SA, no-known-restrictions or US-Government licences; every
pool image is captioned with its Flickr source and licence, and the page
footer credits COCO. `web_demo.py --share` is a different thing: it
tunnels this machine's demo through gradio's public relay (the link is
valid for up to a week) and is meant for short sessions; a permanent
public instance of `web_demo.py` binds to localhost behind the same
reverse proxy. Neither demo has a login: they are public by design.

`./run.sh` needs about 23 GB of GPU memory (observed);
`FAITHFUL=1 ./run.sh` trains the same recipe as 8 × 8 gradient
accumulation for smaller GPUs (see Speed for the small differences
between the two forms). Training writes `train.out` (appended to),
`training_log.json`, `checkpoint.pth`, `model.pth` (the last epoch) and
`best_model.pth` into `$OUT`; the refit adds `best_model_refit.pth`,
`refit.json` and `refit.out`. With the default `OUT` this appends to the
shipped `artifacts/train.out` and replaces the shipped
`training_log.json`, `refit.out`, `refit.json` and
`best_model_refit.pth`; set `OUT` to another directory to keep them.
Environment knobs of `run.sh`: `OUT` (output directory, default
`artifacts`), `WORKERS` (loader workers, default 22, or 8 with
`FAITHFUL`; throughput only — candidate draws are deterministic per
sample; the augmentation's own draws are not seeded, see Speed),
`RETRIES` (attempts before giving up, default 20; each retry resumes
from `$OUT/checkpoint.pth` after 60 s), `FAITHFUL` (any non-empty
value); it runs one OpenMP/MKL/OpenBLAS thread per process. First use
of training downloads torchvision's ImageNet ResNet-18 weights (~47 MB)
and the two torchvision COCO detectors (~92 MB) that generate detector
candidates; the refit, the evaluation
and the demos download no weights. `interactive_demo.py` serves
http://127.0.0.1:7864 (`--host`, `--port`).

## Weights

The released checkpoint is not stored in the repository (about 107 MB)
but hosted in Azure Blob Storage. Download it into the path the commands
above expect, and verify it:

```bash
curl -fL -o artifacts/best_model_refit.pth https://labelqa.blob.core.windows.net/checkpoints/v1.0/paper__artifacts__best_model_refit.pth
sha256sum artifacts/best_model_refit.pth
```

| file | SHA-256 | bytes |
| --- | --- | --- |
| `artifacts/best_model_refit.pth` | `5727521cb7712774df8651e438dc70c1f08a91972bebd6e909d99f486012a9e4` | 107326509 |

Without it, `./run.sh` retrains the model from scratch (160 epochs, about
4 h 25 min on one H100 PCIe 80 GB, as observed, then the refit).

## Results

All numbers below come from one run of `run.sh` (seed 0) plus
`evaluate.py` and `demo_samples.py` on its checkpoint, and each is backed
by a file in `artifacts/`. The paper columns and rows cite its
published numbers (Tables 2 and 4).

**Training and validation** (`artifacts/train.out`,
`artifacts/training_log.json`). All 160 epochs run; the selected
checkpoint is epoch **81**, with validation ROC-AUC **0.879** and accuracy
0.768 at P(good) ≥ 0.5 on the 3 632 test-protocol candidates from the
454 held-out val2017 images. After epoch 81 the validation ROC-AUC eases
to 0.868 at epoch 160 while the accuracy at P(good) ≥ 0.5 falls to 0.678
(from 0.803 at epoch 43): the ranking holds but the raw decision point
drifts, and at the selected epoch it is too strict — before the refit the
checkpoint passes 70% of the good labels (`refit.json`) — which the
held-out refit resets. The checkpoint is selected on this same set, so
0.879 carries the selection's optimism; the test split below is the
independent measurement.

**Held-out refit** (`artifacts/refit.out`, `artifacts/refit.json`), on
7 264 candidates from the same 454 images drawn by the test protocol
(draw rounds 4–11):

| | accuracy | ROC-AUC | good recall | Section 5-weighted accuracy |
| --- | --- | --- | --- | --- |
| before the refit (P(good) ≥ 0.5) | 0.7782 | 0.8839 | 0.700 | 0.7797 |
| decision threshold only, cross-fitted | 0.8095 | — | 0.886 | 0.8113 |
| refit, cross-fitted | **0.8148** | 0.8858 | 0.881 | 0.8171 |

Cross-fitted rows come from fits on one half of the images scored on the
other half, so they are measured on images the fit did not see. The last
column weights the per-subtype recalls with Section 5's subtype
probabilities: the expected accuracy of the test draw. The released
checkpoint uses the fit on all 454 images: weight +1.200 on the fully
connected layers' margin, +0.329 on the head's smooth minimum, +0.341 on
its mean, +0.220 on log(1 + count), intercept +0.914 (in-sample ROC-AUC
0.886). Folded into the checkpoint, the fit makes identical decisions on
all 1 816 candidates of the fold check (largest margin difference
4.7 × 10⁻⁵).

**Test split — accuracy 0.8143, against the paper's 0.8166.**
`evaluate.py` scores the checkpoint on the paper's test protocol (Section
5): each filtered train2017 image contributes one candidate, good or bad
with equal probability, the bad ones drawn with Section 5's subtype
probabilities and the good ones inside the uncertainty regions; a
candidate is called good when P(good) ≥ 0.5 (records
`artifacts/eval.out` and `artifacts/eval.json`, which lists the
arguments: seed 0, `p_exact` 0, batch 32, 22 workers — batch size and
workers change throughput, not the draw; 7.6 minutes on one H100 PCIe
80 GB, as observed). Beside the paper's Table 2:

| | this checkpoint | paper, Table 2 |
| --- | --- | --- |
| class 0 (bad): precision / recall / F1 | 0.856 / 0.756 / 0.802 | 0.86 / 0.75 / 0.80 |
| class 1 (good): precision / recall / F1 | 0.782 / 0.873 / 0.825 | 0.78 / 0.88 / 0.83 |
| macro avg: precision / recall / F1 | 0.819 / 0.814 / 0.814 | 0.82 / 0.82 / 0.82 |
| weighted avg: precision / recall / F1 | 0.819 / 0.814 / 0.814 | 0.82 / 0.82 / 0.82 |
| accuracy | **0.8143** | **0.8166** |
| ROC-AUC | 0.882 | not reported |
| candidates (bad / good) | 108 151 (54 008 / 54 143) | 109 172 (54 473 / 54 699) |

**Table 4, single-error rows.** `evaluate.py` records each bad candidate's
subtype and reports the mean of good-label recall and that subtype's
recall: the expected accuracy of the draw behind Table 4's single-error
rows, where each sample is good or bad with equal probability:

| | A1 | A2 | A3 | B1 | B2 |
| --- | --- | --- | --- | --- | --- |
| this checkpoint | 0.81 | 0.59 | 0.87 | 0.83 | 0.91 |
| paper, Table 4 (mean ± std) | 0.71 ± 0.01 | 0.65 ± 0.02 | 0.71 ± 0.02 | 0.91 ± 0.01 | 0.92 ± 0.01 |

The paper samples 1 000 test images ten times per row; the rows here come
from the whole pool (8 841–13 565 bad candidates per subtype; recall
0.741 / 0.312 / 0.872 / 0.790 / 0.952 against good-label recall 0.873),
which estimates the same quantity with a standard error of about 0.003
or less. Table 4's multi-error combinations (A1A2 … A1A2A3B1B2) are not
measured by `evaluate.py`, and Table 3's small and medium networks are
not trained here.

**Five-sample demo — 5/5** (`python3 demo_samples.py
artifacts/best_model_refit.pth`; record in `artifacts/demo.out`):
0.831 / 0.045 / 0.871 / 0.854 / 0.001 against targets
Good/Bad/Good/Good/Bad. Targets follow one principle: untouched ground
truth = Good, hand-corrupted ground truth = Bad. The `+35%` column
re-scores each sample with every box expanded by 35% — each edge moves
outward by 17.5% of the box dimension; with the drawn boxes clipped to
the image, as the raster draws them, every box stays inside its
image-clipped 0.2 uncertainty region, so the label remains a legal Good
one; it reads
0.684 / 0.067 / 0.857 / 0.833 / 0.002: no verdict changes, which is the
expected behaviour for an in-region perturbation.

**What the comparison establishes.** The protocol is the paper's as this
release reads it: the same splits, error probabilities and balanced draw,
and every Table 2 metric cell within 0.01 of the paper's. Where Section 5
is silent this release chooses: a candidate is called good at
P(good) ≥ 0.5, and the test draw reuses the subtype probabilities that
Section 5 states for training. It is not an identical test set. Two
readings shape what counts as a good label and how hard a bad one is — the
box-scaled region (interpretation 2), and errors applied to a
region-jittered draw of the label, following Figure 12's error examples,
where Section 5's text introduces them to the ground-truth label itself
(`data_loader.corrupt`) — and the pool counts 108 151 candidates against
the paper's 109 172: `load_coco` drops the 1 021 train2017 images that
carry no annotation at all, and since the paper's filter removes only
iscrowd images, its count presumably still includes them. The model is the
paper's method with items 3–12, not the published method alone. Both
accuracies come from single runs and no record here measures run-to-run
variation, so 0.8143 against 0.8166 (a difference of 0.0023) is read as
a match, slightly below the paper's, not an improvement.
Table 4's profile differs: this checkpoint is stronger on
erased boxes (A1) and swapped classes (A3), weaker on distorted boxes
(A2) and added boxes of ground-truth classes (B1), and about level with
the paper on added boxes of absent classes (B2).

## Data

**COCO 2017**: images and `instances_*.json` from
[cocodataset.org](https://cocodataset.org). The model trains on the
val2017 split, as the paper does, with 10% of it held out for validation
and the refit, and is evaluated on candidates from train2017, which
training never sees. `data_loader.load_coco` drops every image that
carries an iscrowd annotation (as the paper's Section 4.2 does) and every
image left with no box: 4 541 val2017 images (the paper counts 4 589,
before the last step) and 108 151 train2017 images. The five demo
photographs are COCO train2017 images — the paper's *test* split,
disjoint from training — fetched once by
`python3 samples/fetch_samples.py` (~1 MB); they are not redistributed
here (three carry NC/ND Flickr licences; see `NOTICE` and
`samples/ATTRIBUTION.md`).

## License

AGPL-3.0-only (`LICENSE`) for the source code and the released checkpoint
`artifacts/best_model_refit.pth` alike; `NOTICE` carries the third-party
attribution for the ImageNet-derived tensors it embeds (torchvision,
BSD-3) and the dataset material.
