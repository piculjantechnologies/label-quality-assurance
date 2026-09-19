# Label quality assurance on Pascal VOC: code for the 2025 PhD thesis

Code for the method of

> Pičuljan, N. *Machine learning-based method for quality assurance of object
> bounding box labels in images.* PhD thesis, University of Zagreb, Faculty of
> Electrical Engineering and Computing, 2025.
> [urn:nbn:hr:168:865107](https://urn.nsk.hr/urn:nbn:hr:168:865107)

with implementation details documented by the thesis's author.
Everything not listed as an implementation detail is the thesis method
unchanged: the uncertainty-region metric (α = β = 0.05, Equation 7.1,
read as the two interpretations state), the eight error types of
Table 8.2 with their 50/50 single-op/composition draw (this code redraws
a bad candidate until the metric labels it bad, and a single op leaves a
label the metric still judges good more often than a composition does,
so the accepted bad candidates lean to compositions), the 10% of bad
draws taken from the two detectors (§8.5 notes that a detector-generated
label's class is not controlled, which can unbalance the support; this
code labels every candidate with the metric), the 224-px inputs and the
label-plane rendering, the augmentation set of Listing 8.11 (its Cutout
read as albumentations' `CoarseDropout` with 1–8 holes of 8–80 px), AdamW
at the thesis hyperparameters, the small-set/large-set split discipline,
and the evaluation protocol.

The implementation details — two interpretations of the printed method
(box-scaled bands, one-to-one matching) and nine items (exact-ground-truth
positives, a correlation fusion in place of the printed single-token
cross-attention, reporting, a per-cell head, a fine-tuned image branch, a
held-out refit, image-hard negatives, image-swap negatives, weight EMA
with selection on validation ROC-AUC; items 1–9) — are listed with the
reason for each under
[Implementation details](#implementation-details-and-why-each-one-is-needed),
after two reported findings; the training settings that depart from
Table 8.4 are tabulated under
[The method, as released](#the-method-as-released), and none of them
changes what counts as a good or bad label.

On the thesis's own large-set protocol the released checkpoint reaches
**0.8913 accuracy and 0.9424 ROC-AUC against the thesis's 0.81 and 0.89**
(Table 8.5, Figure 8.23); [Results](#results) gives the tables and what the
comparison does and does not establish.

## The findings this release reports

**Finding 1: the architecture printed in the thesis cannot use the
image.** As printed in Listings 8.1 and 8.2, both branches are
pooled to a single vector *before* the cross-attention layer, so the
attention sees exactly one key/value token — taken from the label branch,
with the image side supplying only the query. A softmax over one element
is identically 1 whatever the query, so the classifier output depends on
the label branch alone. The graph exports of Figures 8.8–8.10 show the
same single-token network, and the prose of Section 8.3 names it too — a
"cross-attention layer" that "integrates features from the image and
label representations". Item 2 uses a correlation fusion of the two
branches' aligned maps in its place: `CorrNet` in `neural_network.py`,
the network `train.py` builds.

The listings-as-published architecture is kept in `neural_network.py` as
`NetSmall(corr=False)`, next to `NetSmall(corr=True)`, which adds a
correlation term to the same truncated branches; `load_model` loads any
of the three networks, and the released checkpoint is a `CorrNet`.

**Finding 2: the published protocol does not require the image.** The
label statistics of the thesis's error generator carry much of the task.
Seven of its eight error types change a label's box count (erase,
combine, split) or move box edges, often far, outside bands that reach
only 5% of the box dimension either side of each edge — translate shifts
a box by up to its own width and height, resize scales it by 0.04–25 ×,
jitter moves it by up to 112 px, half the input width, crop moves each
edge inward by up to half the box dimension — in ways the label planes
alone expose, while a good candidate never leaves its annotation's
bands; the eighth, swap, moves boxes to a uniformly drawn other class.
Verdicts that *require* reading the image
— a plausible class that is wrong, a box of ordinary size in the wrong
place — are a minority slice of the protocol.

The consequence is that headline accuracy is not diagnostic here: an
accuracy figure on its own says nothing about whether the image was used
at all. The implementation details below make the fusion image-capable
(items 2, 4 and 5), train on labels that only the image can judge
(items 7 and 8) and make the image's contribution measurable (item 3).
On the released model the input ablation measures it directly: with a
mismatched image its ROC-AUC falls from 0.930 to 0.594, and with a blank
image to 0.524, barely above chance ([Results](#results)).

## The method, as released

**Goodness metric.** Each ground-truth box defines a two-sided band per
coordinate — outward by α = 0.05 and inward by β = 0.05 × the box
dimension, clipped to the image (Equation 7.1, box-scaled). A candidate
label is good when every class has exactly as many boxes as regions and
the boxes can be paired with the regions one-to-one, each box inside its
own region's bands (Kuhn's augmenting paths,
`data_loader.is_label_negative`). Every candidate — drawn, corrupted or
detector-produced — is labelled by this metric; image-swap negatives
(item 8) are bad by construction.

**Candidates** (`data_loader.py`). Each training epoch requests one good
and one bad candidate for every training image, drawn anew. One training
sample in ten first passes through the augmentation set of Listing 8.11
(its Cutout read as 1–8 holes of 8–80 px; the ground truth, and the
detector boxes when a detector candidate is drawn, are transformed with the
image; the unaugmented sample is kept when no ground-truth box survives).
Good candidates are the exact ground truth half the time (item 1),
otherwise the thesis draw: every corner inside its band. Bad candidates, in
this order: 10% image-swap (item 8); 10% of the rest from one of two
torchvision COCO detectors (SSDLite320-MobileNetV3-Large or
FasterRCNN-MobileNetV3-Large-320-FPN, drawn per candidate; boxes above
confidence
0.9, COCO classes mapped to VOC); the remainder half from the thesis
generator (the Table 8.2 error types, one op or a composition of ten with
repetition with equal probability, redrawn until the metric rejects the
label) and half image-hard (item 7) — about 10 / 9 / 40 / 40% of the
negatives. The evaluation draw (`evaluate.py`, `analysis.py`'s ablation,
the validation set and the refit) is the thesis generator unchanged: good
candidates by the thesis draw, bad ones 10% from the detectors and 90% from
the error types, no augmentation.

**Architecture** (`neural_network.CorrNet`). Two ResNet-18 branches run
to layer4 (stride 32; 7 × 7 for 224-px inputs). The image branch is
ImageNet-pretrained (torchvision IMAGENET1K_V1) and reads the RGB image,
aspect-preserving resized into 224 × 224 on black padding and normalised
with the ImageNet statistics (`data_loader.normalize_image`); it is
fine-tuned (item 5) with its BatchNorm layers in eval mode, so their
statistics stay the pretrained ones (item 2). The label branch is a
randomly initialised ResNet-18 whose first convolution reads the 20
label planes (one per class; every box drawn as its outline and both
diagonals). The classifier reads both branches' global-average-pooled
maps (512 + 512) and three agreement terms between the aligned stride-32
maps — their element-wise product averaged over the grid, and, on
L2-normalised features, the product averaged over the cells the label
draws on and its per-channel maximum — 5 × 512 = 2 560 features, through
fully connected layers 512 → 256 → 128 → 2 with BatchNorm, ReLU and
dropout 0.5 between them (item 2). The per-cell head (item 4) reads the
image branch's own layer2 (stride 8, 128 channels) and layer3 (stride
16, 256 channels, bilinearly upsampled) next to the label planes
max-pooled 8 × 8 (20 channels) and the class-agnostic drawing
pixel-unshuffled by 8 (64 channels); four 3 × 3 convolution blocks
(dilations 1, 2, 4 and 8, widths 256, 256, 256 and 128, GroupNorm with
16 groups) score, per cell of the 28 × 28 grid, the support for each of
the 20 classes and a 21-way class map; a smooth minimum
(−logsumexp(−v)), the mean and log(1 + count) of the claimed cells'
support enter the good logit through `box_mix`, initialised at zero. The
output is two logits (0 = bad, 1 = good), and P(good) ≥ 0.5 calls a
label good. 26 444 655 parameters: image branch 11 176 512, label branch
11 229 824, classifier 1 477 506, head 2 560 813.

**Training and refit** (`train.py`, `refit.py`, `run.sh` — the recipe's
carrier). `train.py`'s defaults are the recipe except the per-cell head and
its two losses, which `run.sh` adds (`--raster_head --box_weight 0.5
--cls_weight 0.5`) together with the small-set `--pool` from `VOC_POOL`, 20
loader workers (which fix which candidates the shipped run drew), seed 0
(the default, restated) and resuming. AdamW with Table 8.4's values
(learning rate 1 × 10⁻³, betas (0.9, 0.999), eps 1 × 10⁻⁸, weight decay
1 × 10⁻²), the image branch at 0.1 × the rate (1 × 10⁻⁴, item 5); two
linear warmup epochs, then cosine decay to 0 at the last step, stepped per
optimizer step, over 160 epochs; an effective batch of 64 as 32 × 2
gradient accumulation (the 18 requests left over each epoch are dropped,
and the epoch's last optimizer step holds one batch of 32), with BatchNorm
statistics per group of 8 samples (ghost BatchNorm, `--ghost_bn 8`) in
every BatchNorm layer outside the image branch; mixed precision; a per-step
weight EMA of 0.998, whose weights are validated and saved (item 9). The
loss is the verdict's cross-entropy plus 0.5 × the head's per-cell support
loss and 0.5 × its per-cell class loss (item 4). 10% of the small set (582
of 5 823 images) is held out with the run's seed; a fixed validation set of
4 656 candidates is drawn from it once, by the evaluation draw, and scored
every epoch; all 160 epochs run, and `best_model.pth` is the epoch with the
highest validation ROC-AUC (item 9). `refit.py` then fits the verdict on
4 656 other candidates from the same 582 images and writes
`best_model_refit.pth`, the released checkpoint (item 6). The training
settings depart from the thesis as follows; none of them changes what
counts as a good or bad label:

| | Table 8.4 / thesis | `run.sh` |
| --- | --- | --- |
| optimizer | AdamW, the values above; pretrained image branch frozen (Listing 8.1) | the same values; image branch fine-tuned at 0.1 × the rate (item 5) |
| learning rate | not stated | 2 linear warmup epochs, then cosine decay to 0 |
| batch | 4 096 | 64, as 32 × 2 gradient accumulation, BatchNorm statistics per 8 samples |
| epochs | 2 500 | 160, no early stopping; the epoch with the highest validation ROC-AUC is kept |
| augmentation | Listing 8.11 on every training sample | Listing 8.11 on one training sample in ten |
| precision | not stated | mixed precision |
| weight averaging | none | per-step weight EMA 0.998 (item 9) |
| validation and selection | 20% of the small set (§8.4); best validation loss (Listing 8.12) | 10% of the small set (582 images), a fixed evaluation-draw set of 4 656 candidates; best validation ROC-AUC (item 9) |

**Speed.** On one H100 PCIe 80 GB an epoch takes about 37 s (the per-epoch
seconds are in `artifacts/train.out`), so the 160 epochs take about 1 h 40
min, with about 4 GB of GPU memory at batch 32 (as observed). The
engineering measures change the speed, not the distribution a batch is
drawn from: persistent loader workers with pinned memory and non-blocking
host-to-GPU copies; `--prefetch` (batches in flight per worker, default 2);
per-worker caches of the annotations and of each detector's predictions per
image; one OpenMP/MKL/OpenBLAS thread per process (`run.sh`) and one cv2
thread per loader worker, so many workers do not oversubscribe the host;
and `--resume`, which continues an interrupted run from `checkpoint.pth`
(written every epoch) with the optimizer, schedule, gradient scaler, EMA
and main-process RNG state restored — loader workers are re-seeded when a
run resumes, so the continuation is statistical, not bit-exact. Training
candidates are drawn at random by design, so the worker count changes which
candidates are drawn, never their distribution; the validation and refit
candidates are seeded per row and do not depend on it.

## Implementation details, and why each one is needed

**Interpretations (where the thesis's text and listings leave room or
disagree, the reading implemented here):**

- **Box-scaled bands** (`data_loader.build_regions`). The bullets that
  introduce §7.3 give "width and height of the input image", while the
  formal definition sets width = x2 − x1 and height = y2 − y1 of the box,
  as Listing 7.1 does. The bands here scale with the box and are clipped
  to the image.
- **"Exactly one box per region" as a one-to-one matching**
  (`data_loader.is_label_negative`). Equation 7.1 requires that each
  region contain exactly one box; the printed Listing 7.2 counts in-band
  boxes per region, so one box can satisfy two overlapping regions. Here
  a label is good only if each class has as many boxes as regions and the
  boxes can be paired with the regions one-to-one, each box inside its
  own region's bands.

**Items:**

1. **Positives include the exact ground truth** (`P_EXACT_GT = 0.5`,
   `data_loader.good_label`; train-time only — the validation, refit and
   `evaluate.py` draws keep the thesis draw, so evaluation numbers remain
   protocol-comparable). Under the thesis draw every corner is jittered
   inside its band, so the unmodified annotation — the label a user
   actually submits for checking — is drawn only by coincidence: this
   code draws each corner with `random.randint` over its band (the thesis
   states only that good boxes lie within their regions), so exact ground
   truth has positive but vanishing probability once an image has several
   boxes. Drawing it explicitly half the time puts it in the training
   distribution.
2. **Correlation fusion** (`neural_network.CorrNet`) — in place of the
   single-token cross-attention that the printed listings and figures
   show and the prose names (finding 1). Listing 8.1 truncates both
   ResNet-18 branches after their first residual stage and pools each to
   one vector before the two meet. Here both branches run to layer4, and
   the network compares their aligned stride-32 maps position by
   position: the element-wise product averaged over the grid and, on
   L2-normalised features, the product averaged over the cells the label
   draws on and its per-channel maximum join the two pooled branch
   vectors in the classifier input (2 560 features, fully connected
   layers 512 → 256 → 128 → 2), so the verdict can ask whether image
   evidence appears where the label claims an object. Two input-side
   details are bundled inside this item because they only become live
   with it: images are fed to the pretrained branch in RGB with the
   standard ImageNet statistics — the convention its weights expect (one
   exception: Listing 8.11's `ChannelShuffle`, p = 0.5, is kept; with the
   augmentations applied to one sample in ten it reaches about 5% of
   training images) — and the image branch's BatchNorm layers stay in
   eval mode during training (also while item 5 fine-tunes its weights),
   so their running statistics do not drift from the pretrained ones.
3. **Reporting** (`analysis.py`): the input ablation
   (true/mismatched/blank — the direct grounding measure on a protocol
   whose label statistics carry much of the task, finding 2) and
   bad-recall per thesis error type (Table 8.2 ops, single-op
   candidates).
4. **Per-cell label-plane head** (`neural_network.CorrNet`,
   `--raster_head --box_weight 0.5 --cls_weight 0.5`, which `run.sh`
   passes). Equation 7.1 rejects a label when *any* one of its boxes
   leaves its bands, but both branches are pooled to a vector before the
   classifier, and item 2's agreement terms are pooled over the grid as
   well: one box out of band, among several in band, is averaged away.
   The head judges the label where it is drawn. On the stride-8 grid
   (28 × 28 cells for 224-px inputs) it reads the image branch's own
   semantic features — its layer2 (stride 8) and layer3 (stride 16,
   upsampled) — next to the label planes themselves: which classes are
   drawn in each cell (every plane max-pooled 8 × 8) and where inside the
   cell they run (the class-agnostic drawing, pixel-unshuffled into 64
   channels, so a shift of one pixel is still visible). Four 3 × 3
   convolution blocks with dilations 1, 2, 4 and 8 widen each cell's view
   around a box edge; per cell the head scores the support for each of
   the 20 classes and, as a training signal, the class the image shows
   there (20 = nothing). The label's own planes select the claimed
   classes' support, and the verdict adds a smooth minimum and the mean
   of the claimed cells' support and log(1 + their count) to the good
   logit through a layer initialised at zero, so the network starts as
   exactly the item-2 fusion. Two per-cell losses, each weighted 0.5,
   train it next to the verdict's cross-entropy
   (`data_loader.cell_targets`): each claimed cell's support against the
   per-box form of Equation 7.1 — band membership for the box whose
   drawing covers the cell, the minimum where several do — and every
   cell's class against the ground-truth class map. The model's inputs
   are unchanged (image and label planes), and at inference it returns
   the two verdict logits. The head adds 2 560 813 trained parameters;
   the whole model has 26 444 655. What its per-cell target cannot flag
   is a count error: a box that is missing leaves no cell to score, and a
   surplus box that lies inside some band scores as good there, so labels
   the metric rejects for the box count or the one-to-one matching are
   left to the fusion.
5. **Fine-tuned image branch** (`--img_lr_mult 0.1`, the `train.py`
   default). Table 8.4 uses an ImageNet-pretrained ResNet-18 for the
   image, and Listing 8.1 freezes it. ImageNet features are not tuned to
   whether a drawn edge sits on an object's boundary, which is what items
   2 and 4 ask of them, so the whole image branch — conv1 to layer4,
   including the layer2 and layer3 the head reads — trains here at a
   tenth of the learning rate (1 × 10⁻⁴): the pretrained features are
   adapted rather than overwritten. Its BatchNorm statistics stay frozen
   (item 2).
6. **Held-out refit** (`refit.py`, the last step of `run.sh`). The
   verdict's margin is a weighted sum of the fully connected layers' margin
   and the head's three terms, plus a bias. Training fits these weights on
   images the model has partly memorised, and on a candidate mix (exact
   ground truth, image-hard, image-swap and augmented candidates) that
   differs from the evaluation draw; on unseen images the margin shifts and
   the decision point moves with it — before the refit the checkpoint calls
   too many good labels bad (good recall 0.602 against bad recall 0.951 on
   held-out candidates, [Results](#results)). `refit.py` refits the five
   numbers by logistic regression on held-out data — the run's own
   validation images (the 10% of the small set that training never trains
   on, 582 images), with candidates drawn as `evaluate.py` draws them, four
   per image and requested label from draw rounds 4–7, which never overlap
   the validation set's rounds 0–3 (4 656 candidates) — and folds them into
   the last fully connected layer and the head's mixing layer, so the
   released checkpoint is an ordinary model with the same two inputs. The
   large set is never used. Most of the effect is a shift of the decision
   threshold; the rest re-weights the head against the fully connected
   layers.
7. **Image-hard negatives** (`data_loader.corrupt_mixed`,
   `--ih_share 0.5`). Of the generated negatives — those that are neither
   image-swap nor detector candidates — half come from the thesis
   generator and half are image-hard: a good draw with one error of a
   kind the label's geometry does not give away — one box deleted (on
   images with two or more boxes), one box relabelled to an absent class
   drawn by how often it co-occurs with the present ones, one box
   relocated at its own size to a random place, or 1–3 real ground-truth
   boxes added from a per-class bank, of a present class or of a
   co-occurrence-plausible absent class — kept only if the metric rejects
   it (30 tries, else the thesis generator). The bank and the
   co-occurrence table are built from the training images only
   (`data_loader.build_context`). The thesis generator's corruptions are
   mostly visible in the label itself (finding 2); a label whose boxes
   have ordinary sizes, counts and classes but disagree with the picture
   can only be judged by reading the image, and these negatives put such
   labels into training.
8. **Image-swap negatives** (`data_loader.Data`, `--swap_share 0.1`). A
   tenth of the bad requests pair a good label of the image (drawn as
   item 1 draws positives) with another training image, stretched to
   this image's size, and are labelled bad; their per-cell targets are
   all bad and their class map is the other image's ground truth. A label
   that is right for its own image is wrong for any other, and no
   statistic of the label can reveal that — only the image can — so this
   negative trains the verdict to depend on the image.
9. **Weight EMA, and selection by validation ROC-AUC on a fixed evaluation
   draw** (`--ema_decay 0.998`, `--val_split 0.1`, `--val_rounds 4`,
   `data_loader.fixed_rows`). A per-step exponential moving average of the
   weights is kept, averaging about the last 500 optimizer steps (about
   three epochs of 164 steps); validation and the saved models
   (`best_model.pth`, `model.pth`) use the EMA weights, so every validation
   number reported here is an EMA-weights number. The average damps the
   step-to-step noise of the weights among which validation selects. The
   validation set — 10% of the small set, 582 images — is drawn once as
   `evaluate.py` draws the large set (draw rounds 0–3, four candidates per
   image and requested label, 4 656 candidates) and is the same every
   epoch, so it measures the task the model is evaluated on rather than the
   training mix, and its epoch-to-epoch differences are the model's, not
   the draw's. Selection keeps the epoch with the highest validation
   ROC-AUC: the decision threshold is set afterwards by the refit (item 6),
   so what selection should reward is the ranking, which ROC-AUC measures
   and the validation loss mixes with calibration. In the shipped run the
   two disagree: validation loss is lowest at epoch 19 (0.297), validation
   ROC-AUC highest at epoch 84 (0.9411; `artifacts/train.out`).

`train.py` keeps options outside the recipe: `--patience N` stops after N
epochs without a validation ROC-AUC improvement (0, the recipe, trains all
epochs), `--no_amp` trains in fp32, and `--p_exact_gt 0 --ih_share 0
--swap_share 0` trains on the thesis's own candidate draw (items 1, 7 and 8
off; the augmentation still runs on one sample in ten). The shipped run
leaves them at their defaults (`training_log.json` records every argument).

## Layout

| file | role |
| --- | --- |
| `data_loader.py` | pools, regions, metric, generator, rendering, augmentation, the evaluation draw and the fixed validation and refit rows (items 1, 2, 3, 4, 6, 7, 8, 9) |
| `neural_network.py` | `CorrNet`, the released network (items 2, 4, 5); the listings' `NetSmall` (`corr=False`, finding 1) and its correlation variant (`corr=True`); ghost BatchNorm; `load_model` |
| `train.py`, `run.sh` | the trainer, whose defaults are the recipe except item 4's head (items 1, 5, 7, 8, 9), and the recipe carrier (adds item 4's head, runs item 6's refit) |
| `refit.py` | the held-out refit of the verdict layer (item 6) |
| `evaluate.py` | the unchanged large-set protocol (loads any of the three networks) |
| `analysis.py` | item-3 measurement suite (input ablation, bad-recall per error type) |
| `interactive_demo.py` + `interactive_demo.html` | box editor with live model-vs-metric verdicts; also scores your own uploaded photos |
| `prepare_voc_data.py`, `download_pascal_voc_2012_dataset.py` | VOC data preparation |
| `artifacts/` | the shipped run: `train.out`, `training_log.json`, `refit.out`, `refit.json`, `eval.out`, `eval.json`, `analysis_release.json` (`best_model_refit.pth` is downloaded separately, see Weights) |
| `requirements.txt`, `LICENSE`, `NOTICE` | packaging |

## Setup

```bash
pip install -r requirements.txt

# fetch the fiftyone export (~/fiftyone/voc-2012/{train,validation}/
# {data,labels.json}) and pack each split's annotations once
python3 download_pascal_voc_2012_dataset.py
python3 prepare_voc_data.py --split validation   # the small set (trains)
python3 prepare_voc_data.py --split train        # the large set (held out)
```

`prepare_voc_data.py` writes `processed_annotations/` next to each split's
`data/`; pass `--dataset_dir` if the export lives elsewhere. `run.sh`
then reads the small set from `VOC_POOL`, a split directory that contains
both `processed_annotations/` and `data/` (default
`~/fiftyone/voc-2012/validation`); the Python scripts take the
`processed_annotations/` directory itself as `--pool` (`train.py` and
`refit.py` default to the validation split's, `evaluate.py`,
`analysis.py` and `interactive_demo.py` to the train split's).

## Run

```bash
./run.sh      # trains on the small set into artifacts/ (resumable), then
              # refit.py writes artifacts/best_model_refit.pth, the
              # released checkpoint

# the two steps of run.sh by hand (run.sh also passes --pool from VOC_POOL)
python3 train.py --raster_head --box_weight 0.5 --cls_weight 0.5 \
    --workers 20 --seed 0 --out artifacts --resume
python3 refit.py artifacts/best_model.pth --seed 0 \
    --out artifacts/best_model_refit.pth --report artifacts/refit.json

# evaluate on the unchanged large-set protocol (its printout is what
# artifacts/eval.out holds):
python3 evaluate.py --ckpt artifacts/best_model_refit.pth --out artifacts/eval.json
# the measurement suite (--out writes the json the Results section cites):
python3 analysis.py --ablation --per_op \
    --ckpt artifacts/best_model_refit.pth --out artifacts/analysis_release.json

python3 interactive_demo.py     # box editor at 127.0.0.1:7863 — drag/resize/
                                # relabel boxes on held-out images; the model's
                                # P(good) and the Eq 7.1 metric respond to every
                                # edit, with the uncertainty bands drawn on the
                                # image.
python3 interactive_demo.py --uploads-only
                                # the same editor for your own photos only; runs
                                # on the downloaded checkpoint, no VOC needed.
```

In the editor, "upload photo…" (or dropping an image on the page) scores
your own photo: draw and edit boxes with the 20 VOC classes, and the
model's P(good) follows every edit. A photo has no ground truth, so the
metric column reads n/a and only the model judges it. A photo larger than
VOC's scale is area-downscaled to a longest side of 500 px first, then
takes the pipeline's own resize to 224 px, as every training image does.
The page accepts files up to 8 MB. Uploaded photos are held only in server
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
trailing-slash URL — the page uses relative `api/` URLs. Every pool
image is captioned as a Pascal VOC 2012 Flickr photograph under the VOC
terms of use, and the page footer credits the dataset. There is no
login: the demo is public by design.

`./run.sh` needs about 4 GB of GPU memory (at batch 32, as observed).
Training writes `train.out` (appended to), `training_log.json`,
`checkpoint.pth`, `model.pth` (the last epoch) and `best_model.pth` into
`$OUT`; the refit adds `best_model_refit.pth`, `refit.json` and
`refit.out`. With the default `OUT` this appends to the shipped
`artifacts/train.out` and replaces the shipped `training_log.json`,
`refit.out`, `refit.json` and `best_model_refit.pth`; set `OUT` to another
directory to keep them. Environment knobs of `run.sh`: `VOC_POOL` (the
small-set split directory, default `~/fiftyone/voc-2012/validation`), `OUT`
(output directory, default `artifacts`), `WORKERS` (loader workers, default
20; throughput, and which training candidates are drawn, not their
distribution), `RETRIES` (attempts before giving up, default 20; each retry
resumes from `$OUT/checkpoint.pth` after 60 s); it runs one
OpenMP/MKL/OpenBLAS thread per process. First use of `train.py`,
`refit.py`, `evaluate.py` or `analysis.py` downloads the two torchvision
COCO detectors (~92 MB) that generate detector candidates; training
additionally downloads the ImageNet ResNet-18 weights (~47 MB).
`interactive_demo.py` serves http://127.0.0.1:7863 (`--host`, `--port`;
`--split validation` browses the training pool instead of the held-out
large set).

## Weights

The released checkpoint is not stored in the repository (about 106 MB)
but hosted in Azure Blob Storage. Download it into the path the commands
above expect, and verify it:

```bash
curl -fL -o artifacts/best_model_refit.pth https://labelqa.blob.core.windows.net/checkpoints/v1.0/thesis__artifacts__best_model_refit.pth
sha256sum artifacts/best_model_refit.pth
```

| file | SHA-256 | bytes |
| --- | --- | --- |
| `artifacts/best_model_refit.pth` | `27040e830f3f927a6f4a797ff857140e5b33826a2129c74dacc60ca30f1590ca` | 105951820 |

Without it, `./run.sh` retrains the model from scratch (160 epochs,
about 1 h 40 min on one H100 PCIe 80 GB, as observed, then the refit).

## Results

All numbers below come from one run of `run.sh` (seed 0: 160 epochs of
training, then the held-out refit) and the evaluation and measurement
commands of [Run](#run), and each is backed by a file in `artifacts/`.
The thesis columns and rows cite its published numbers (Tables 8.5 and
8.6, Figure 8.23).

**Training and validation** (`artifacts/train.out`,
`artifacts/training_log.json`). All 160 epochs run; the selected checkpoint
is epoch **84**, with validation ROC-AUC **0.9411**, accuracy 0.771 at
P(good) ≥ 0.5 and loss 0.703 on the 4 656 candidates drawn from the 582
held-out small-set images. Validation ROC-AUC declines slowly after that
epoch (0.9285 at epoch 160, loss 1.556), and selection keeps epoch 84. The
low accuracy at P(good) ≥ 0.5 is the too-strict decision point that the
refit moves. The checkpoint is selected on this same set, so 0.9411 carries
the selection's optimism; the large-set evaluation below is the independent
measurement.

**Held-out refit** (`artifacts/refit.out`, `artifacts/refit.json`), on
4 656 candidates (2 369 good) from the same 582 images, drawn as
`evaluate.py` draws them (draw rounds 4–7):

| | accuracy | ROC-AUC | good recall | bad recall |
| --- | --- | --- | --- | --- |
| before the refit (P(good) ≥ 0.5) | 0.7738 | 0.9445 | 0.602 | 0.951 |
| decision threshold only, cross-fitted | 0.8902 | — | 0.920 | 0.860 |
| refit, cross-fitted | **0.8969** | 0.9469 | 0.930 | 0.862 |

Cross-fitted rows come from fits on one half of the images scored on the
other half, so they are measured on images the fit did not see. The
released checkpoint uses the fit on all 582 images: weight +1.109 on the
fully connected layers' margin, +0.110 on the head's smooth minimum, +0.646
on its mean, −0.168 on log(1 + count), intercept +2.475 (in-sample ROC-AUC
0.948). Folded into the checkpoint, the fit makes identical decisions on
all 1 024 candidates of the fold check (largest margin difference
2.2 × 10⁻⁶).

**Large-set evaluation** (the unchanged thesis protocol, 5 717 images /
11 434 candidates; `artifacts/eval.out`, `artifacts/eval.json`):

| | this checkpoint | thesis, Table 8.5 |
| --- | --- | --- |
| class 0 (bad): precision / recall / F1 | 0.911 / 0.863 / 0.886 | 0.86 / 0.74 / 0.80 |
| class 1 (good): precision / recall / F1 | 0.874 / 0.919 / 0.896 | 0.78 / 0.88 / 0.82 |
| weighted F1 | 0.891 | 0.81 |
| accuracy | **0.8913** | **0.81** (0.8124 from Table 8.6) |
| ROC-AUC | **0.9424** | 0.89 (Figure 8.23) |
| confusion matrix TN / FP / FN / TP | 4 841 / 770 / 473 / 5 350 | 4 256 / 1 461 / 684 / 5 033 (Table 8.6) |
| candidates (bad / good) | 11 434 (5 611 / 5 823) | 11 434 (5 717 / 5 717) |

The release is evaluated on a seeded draw of the error generator
(`evaluate.py --seed 0`, 4 loader workers, batch 256; the draw depends on
these three and not on the network, which is built before seeding),
distinct from the thesis's own. The support differs because this code
labels every candidate with the metric, so a detector draw requested as
bad can turn out good; the thesis table's equal support (5 717 / 5 717)
suggests it counts the requested classes.

**Input ablation** (`analysis.py --ablation`, record in
`artifacts/analysis_release.json`): 1 000 candidates — one bad-requested
and one good-requested from each of 500 large-set images — each scored
three times: with the true image, with a mismatched image (the batch's
images rolled by two, so no candidate gets its own image back) and with a
blank image:

| condition | accuracy | ROC-AUC |
| --- | --- | --- |
| true image | 0.883 | 0.930 |
| mismatched image | 0.495 | 0.594 |
| blank image | 0.461 | 0.524 |

Without the right image the verdict collapses: accuracy falls to about
chance under both degraded conditions, and of the discriminative margin
over chance (ROC-AUC − 0.5) 22% survives with a mismatched image and 6%
with a blank one, so the verdict is carried by the image and the label
together, not by the label alone (the largest change in P(good) against
the true image is 0.989; a model whose verdicts came from the label
branch alone would score all three conditions identically). One
qualification keeps this honest: the blank condition is off-distribution,
so it overstates what a merely uninformative image would cost.

**Bad-recall per thesis error type** (`analysis.py --per_op`, single-op
candidates from the large set, n = 150 each; same file): swap 0.86, split
0.86, resize 0.84, jitter 0.84, translate 0.78, crop 0.76, erase 0.59,
combine 0.33 (binomial SE 0.03–0.04, so most neighbouring rows are not
distinguishable; combine and erase are clearly the weakest). Errors that
relabel, split, reshape or move a box are caught most often; the hardest
are combine and erase — the ops that merge two or more boxes into one or
remove boxes, which leave the remaining drawing consistent with a plausible
label.

The erase row averages two populations that behave nothing alike, and
`analysis.py` reports them separately: when the op removes *every* box
the label is rejected every time (n = 58, bad-recall 1.00), and when
boxes remain it is caught in 33% of cases (n = 92) — with combine (0.33)
the weakest results, and the ones a reader should take as the method's
limit on missing and merged boxes: the per-cell head scores boxes that
are drawn, and a box that is not drawn leaves no cell to score (item 4).
The thesis reports these operations only in aggregate (Table 8.5); this
breakdown is reported as measured.

**What the comparison establishes.** Single seed (the evaluation draw is
stochastic, so small-digit differences are noise). The release's margin
over the thesis is +7.9 points: Table 8.5 prints 0.81, and the confusion
matrix of Table 8.6 gives 9 289 / 11 434 = 0.8124 against the release's
0.8913 (the release accuracy's own standard error is 0.3 points). The
large-set, ablation and per-op numbers are P(good) ≥ 0.5 decisions on
candidates from the large set (the VOC train split), held out from
training, which uses the small set; the training and refit numbers are on
the small set's held-out validation images, which also selected the epoch
(the refit draws its own candidates from them). The comparison measures
this implementation — the thesis method with items 1–9 and the training
settings above — against the published numbers. No per-item ablation was
run, so no single number can be credited to one item: the refit's own share
is measured above (cross-fitted, 0.7738 → 0.8969 on the validation
candidates, most of it the threshold), and the other items and the settings
are not separated; the item-2 bundle is additionally inseparable by
construction (the RGB and BatchNorm handling only become live with the
fusion). `P_EXACT_GT` and the image-hard and image-swap negatives apply to
training only, so the evaluation rows above are protocol-comparable.

## Data

**Pascal VOC 2012**: `download_pascal_voc_2012_dataset.py` fetches the
dataset through the fiftyone zoo; `prepare_voc_data.py` packs the
per-image annotations. The validation split (5 823 images) is the
thesis's small set: 90% trains, and 10% (582 images) is held out for
selection and the refit. The train split (5 717 images) is the large
set, used only by the evaluation, the measurement suite and the demo's
default pool. No VOC images or annotations are redistributed (see
`NOTICE`).

## License

AGPL-3.0-only (`LICENSE`) for the source code and the released checkpoint
`artifacts/best_model_refit.pth` alike; `NOTICE` carries the third-party
attribution for the ImageNet-derived tensors it embeds (torchvision,
BSD-3) and the dataset material.
