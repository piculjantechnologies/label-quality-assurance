# Unified label quality assurance for object bounding boxes

One implementation of the label-quality-assurance task studied in

> Pičuljan, N. and Car, Ž. *Machine Learning-Based Label Quality Assurance for
> Object Detection Projects in Requirements Engineering.* Applied Sciences
> 13(10):6234, 2023. [doi:10.3390/app13106234](https://doi.org/10.3390/app13106234)
>
> Pičuljan, N. *Machine learning-based method for quality assurance of object
> bounding box labels in images.* PhD thesis, University of Zagreb, Faculty of
> Electrical Engineering and Computing, 2025.
> [urn:nbn:hr:168:865107](https://urn.nsk.hr/urn:nbn:hr:168:865107)

written once by the author of both, dataset-parameterised: every file that
defines the method — the goodness metric, the candidate generators, the
model, the trainer, the calibration — is dataset-agnostic, and `dataset.py`
selects a configuration by environment variable:

    QA_DATASET=coco     COCO 2017, 80 classes, 640-px inputs
    QA_DATASET=voc      Pascal VOC 2012, 20 classes, 224-px inputs

Its implementation details — five items with respect to the paper (region
definition, loss on logits, exact-ground-truth positives, image
grounding, structured supervision) and four with respect to the thesis
(spatial fusion in place of the printed cross-attention,
exact-ground-truth and edge-biased positives, the negative mixer,
measurement) — are listed with the reason for each under
[Implementation details](#implementation-details-and-why-each-one-is-needed).

The released models reach a validation ROC-AUC of **0.940 on VOC and
0.930 on COCO**, and on the 5 717 held-out images of the VOC large set,
under the paper's error taxonomy, the VOC model reaches **0.857 accuracy
and 0.916 ROC-AUC**; [Results](#results) gives the tables and what they
do and do not establish.

## The method, as released

**Goodness metric.** Each ground-truth box defines a two-sided band per
coordinate — outward and inward by 0.05 × the box dimension. A candidate
label is good when every class has exactly as many boxes as regions and a
perfect region-to-box matching exists with every matched box in-band
(Kuhn's algorithm). Every candidate — drawn, corrupted, or
detector-produced — is labelled by this metric.

**Candidates** (`qa_data.py`). Good draws place every corner inside its
band: the exact ground truth verbatim half the time, and in the jittered
half each box is drawn, with probability 0.25, inside the boundary
quarter of its band — nearest either the outward or the inward edge —
the hardest true-positive region. Negatives come from one mixer, drawn
in this order: 10% pair a metric-good label with a *different* pool
image — the grounding negative that label geometry alone can never
reveal; 10% of the rest are **detector** outputs
(SSDLite320-MobileNetV3-Large / FasterRCNN-MobileNetV3-Large-320-FPN
above confidence 0.9); the remainder splits evenly between
**image-hard** errors (a single deletion, a co-occurrence-plausible class
swap, an off-region relocation, and real boxes of present or
plausible-absent classes from a bank over the training images) and
**severity-controlled** coordinate/structural corruptions — about
10 / 9 / 40 / 40% overall. Half of the image-hard and severity negatives
(about 40% of all negatives) stack 1–3 extra error types.

**Architecture** (`qa_model.py`). Two trunks — the image trunk an
ImageNet-pretrained ResNet-50 frozen through `layer2` including its
BatchNorm statistics, the label trunk a ResNet-18 from scratch reading
one rendered plane per class — fused spatially as
`[image, label, image × label]`, pooled by average, maximum and
labelled-region mean, plus a per-class claimed-position agreement vector
(supervised by an auxiliary box-class loss) with its lower-quartile
worst-box variant, and a **per-box verification head**: every claimed box
is scored alone from RoI features over the fused grid, a 2× context
window and its claimed-class evidence channel, outputting a support logit and a
signed band-margin regression; the sample head sees the per-box minimum
and mean — the differentiable form of the metric's "bad if ANY box is
bad". A per-class count head is available (`--count_weight`) and is off
in the released recipe.

**Training and calibration** (`qa_train.py`, `qa_calibrate.py`,
`run_qa.sh` — the recipe's only carrier). AdamW (learning rate
1 × 10⁻³, weight decay 1 × 10⁻²), cosine schedule with 2 warmup epochs,
an effective batch of 64 with BatchNorm statistics per group of 8
samples (ghost BatchNorm, `--ghost_bn 8`), mixed precision, 60 epochs,
seed 0; the image trunk's unfrozen stages (`layer3`/`layer4`) train at
0.1 × the base learning rate, and flip/affine/photometric augmentation
is applied to half of the training samples. A per-step weight EMA of
0.998 is tracked next to the raw weights and is the released candidate
on both datasets (raw and EMA validation ROC-AUC tie within 0.0002); it
is calibrated by folding the validation-optimal margin into the final
bias (validation data only, ROC-AUC unchanged up to
half-precision rounding). The released decision model per dataset is
`artifacts_<dataset>/best_ema_calibrated.pth`.

**Speed.** The released recipe's effective batch is 64 with BatchNorm
statistics per group of 8 samples, and `./run_qa.sh` runs it as one
physical batch of 64: `GhostBatchNorm` computes the statistics per virtual
sub-batch of 8, which reproduces the accumulated recipe's effective batch
and image-level BatchNorm statistics (about 30 GB of GPU memory, observed;
`FAITHFUL=1` runs the accumulated form for small GPUs). The two forms are
not bit-identical: the auxiliary, per-box and margin terms are averaged
over the whole batch of 64 rather than per micro-batch, the box head's
BatchNorm1d normalises over all box rows of the 64 samples (in groups of 8
rows only when their count is a multiple of 8) rather than over each
micro-batch's rows, and the accumulated form takes one extra partial step
per epoch. Sparse plane transport ships only the class planes that hold
boxes from the loader workers and scatters the dense raster on the GPU
(the dense raster is mostly zeros) — bit-identical batches. Loader workers
persist across epochs and cache decoded images (`--cache_images`, at a
host-RAM cost of roughly 3.6 GiB per worker for COCO and 0.5 GiB for VOC),
which changes scheduling only: the candidate draws — good/bad choice,
corruption, detector and swap selection — are a function of (seed, epoch,
index); the photometric and geometric augmentation itself is unseeded, so
a re-run reproduces the recipe in distribution, not bit for bit.

## Implementation details, and why each one is needed

**With respect to the paper (2023):**

1. **Region definition** — the thesis's two-sided box-scaled band is used
   in place of the paper's 0.2 outward-only region, whose text reads it
   as image-scaled while its figures draw it box-scaled; the band follows
   the box-scaled reading and makes the positive region a property of the
   box rather than of the image.
2. **Loss on logits** — cross-entropy is applied to logits, as the paper's
   text specifies, where its figures end in a sigmoid.
3. **Exact-ground-truth positives** — the label a user actually submits is
   drawn verbatim half the time; under the paper's continuous draw it is
   a measure-zero event.
4. **Image grounding** — the paper's pool-then-concat fusion pools each
   branch before the two meet, so no layer can compare a box with the
   image content it claims. The spatial fusion, the image-hard and
   image-swap negatives, and the auxiliary agreement supervision are the
   recipe's answer; together they make the verdict depend on the image
   (no per-component ablation was run, so no single element is
   credited): with the image blanked the released models fall to
   0.55–0.58 ROC-AUC, barely above chance, so the label planes alone do
   not carry the task ([Results](#results)).
5. **Structured supervision** — the per-box head scores each claimed box
   alone (support logit and band margin) and feeds the sample verdict its
   minimum and mean.

**With respect to the thesis (2025):**

1. **Spatial fusion in place of the printed cross-attention** — the
   printed Listings 8.1/8.2 pool both branches to single vectors before
   the cross-attention, so attention runs over one token and the
   classifier depends on the label branch alone; the printed figures
   show the same single-token network, and the prose of §8.3 names it.
   This implementation fuses the two branches' aligned maps spatially in
   its place (fusion before pooling, at scale); it is not a copy of the
   printed network.
2. **Exact-ground-truth and edge-biased positives** (as above; the thesis
   draw jitters every corner).
3. **The negative mixer** — the thesis's eight error types (Table 8.2)
   are covered by severity-controlled coordinate and structural ops inside
   a broader mixer with detector, image-hard, composition and image-swap
   negatives, closing the label-statistics shortcut the thesis protocol
   permits.
4. **Measurement** — the input ablation (true/mismatched/blank) ships with
   the release, making image grounding a measured property instead of an
   assumption; the per-box head's support and margin outputs are exposed
   through `FusionNet.forward(..., return_aux=True)` (no shipped tool
   prints them).

The metric is the thesis band, applied to both datasets (item 1 on the
paper side); the small-set/large-set discipline, the class lists, the
input resolutions and the evaluation pools follow the two sources
unchanged. One implementation rule is in neither publication: a
ground-truth box thinner than 2 px after resizing cannot be rasterised
and is dropped (widened instead when the image would otherwise hold no
box).

## Layout

| file | role |
| --- | --- |
| `qa_data.py` | metric, candidates, mixer, rendering, augmentation |
| `qa_model.py` | the released architecture (+ GhostBatchNorm) |
| `qa_train.py`, `run_qa.sh` | the trainer and the recipe carrier |
| `qa_calibrate.py` | validation-only decision calibration |
| `qa_ablation.py` | input ablation (true / mismatched / blank image) |
| `qa_protocol.py` | band-metric evaluation over a held-out pool (31 error combinations) |
| `qa_demo.py`, `samples/` | the five-sample demo (COCO configuration) |
| `qa_interactive_demo.py` + `interactive_demo.html` | box editor with live model-vs-metric verdicts (both configurations); also scores your own uploaded photos |
| `dataset.py`, `dataset_coco.py`, `dataset_voc.py` | dataset selection and the two configurations |
| `prepare_voc_data.py`, `download_pascal_voc_2012_dataset.py` | VOC data preparation |
| `artifacts_voc/` | the shipped VOC run: `train.out`, `training_log.json`, `calibrate.out`, `ablation.out`, `protocol.out`, `protocol.json` (`best_ema_calibrated.pth` is downloaded separately, see Weights) |
| `artifacts_coco/` | the shipped COCO run: `train.out`, `training_log.json`, `calibrate.out`, `ablation.out`, `demo.out` (`best_ema_calibrated.pth` is downloaded separately, see Weights) |
| `requirements.txt`, `LICENSE`, `NOTICE` | packaging |

## Setup

```bash
pip install -r requirements.txt

# VOC: fetch the fiftyone export (~/fiftyone/voc-2012/{train,validation}/
# {data,labels.json}) and pack each split's annotations once
python3 download_pascal_voc_2012_dataset.py
python3 prepare_voc_data.py --split validation   # the training pool
python3 prepare_voc_data.py --split train        # the held-out demo/protocol pool
```

`prepare_voc_data.py` writes `processed_annotations/` next to each split's
`data/`; pass `--dataset_dir` if the export lives elsewhere. `VOC_POOL`
must name a split directory that contains both `processed_annotations/`
and `data/`. With the default export location the training and
calibration scripts can do without it for the validation split
(`dataset_voc.py` defaults to `~/fiftyone/voc-2012/validation`);
`qa_interactive_demo.py` serves VOC only when `VOC_POOL` is set or
`--datasets voc` is passed. For the held-out demo/protocol pool
`VOC_POOL` names the `train` directory instead. COCO needs no
preparation step: `COCO_IMAGES` names the raw image directory and
`COCO_ANNOTATIONS` the matching `instances_*.json`.

## Run

```bash
# VOC configuration
export QA_DATASET=voc VOC_POOL=/path/to/voc-2012/validation
./run_qa.sh                              # trains into artifacts_voc/ (resumable)
python3 qa_calibrate.py artifacts_voc/best_ema.pth \
    --out artifacts_voc/best_ema_calibrated.pth \
    --p_edge 0.25 --swap_share 0.1
QA_DATASET=voc VOC_POOL=/path/to/voc-2012/train \
    python3 qa_protocol.py artifacts_voc/best_ema_calibrated.pth \
    --out artifacts_voc/protocol.json
python3 qa_ablation.py artifacts_voc/best_ema_calibrated.pth \
    --p_edge 0.25 --swap_share 0.1
VOC_POOL=/path/to/voc-2012/train \
    python3 qa_interactive_demo.py --datasets voc    # the held-out large set

# COCO configuration
export QA_DATASET=coco COCO_IMAGES=/path/to/val2017 \
       COCO_ANNOTATIONS=/path/to/instances_val2017.json
./run_qa.sh
python3 qa_calibrate.py artifacts_coco/best_ema.pth \
    --out artifacts_coco/best_ema_calibrated.pth \
    --p_edge 0.25 --swap_share 0.1
python3 qa_ablation.py artifacts_coco/best_ema_calibrated.pth \
    --p_edge 0.25 --swap_share 0.1
python3 samples/fetch_samples.py                # one-time demo images
python3 qa_demo.py artifacts_coco/best_ema_calibrated.pth
python3 qa_interactive_demo.py --datasets coco --heldout coco   # the 5% split never trained on

# your own photos only: no dataset needed, every checkpoint present is served
python3 qa_interactive_demo.py --uploads-only
```

The demo serves every dataset whose data variables are set; `--datasets`
restricts it to one.

In the editor, "upload photo…" (or dropping an image on the page) scores
your own photo with the selected configuration's model: draw and edit
boxes with its classes, and the model's P(good) follows every edit. A
photo has no ground truth, so the metric column reads n/a and only the
model judges it. A photo larger than the dataset's own image scale
(longest side 640 px for COCO, 500 px for VOC) is area-downscaled to it
first; the model then letterboxes it like any training image. Switching
the dataset selector on a photo re-scores the same photo with the other
configuration's model, carrying the boxes over (the six classes the two
datasets spell differently are renamed; boxes with no counterpart are
dropped, and the page says how many). The page accepts files up to 8 MB.
Uploaded photos are held only in server memory for re-scoring — never
written to disk or logged — and are discarded 30 minutes after they were
last scored, when 64 more recently used photos are held, or when the
server restarts.

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
CC BY, CC BY-SA, no-known-restrictions or US-Government licences (the
VOC pool is unaffected); every pool image is captioned with its source
and licence, and the page footer credits both datasets. There is no
login: the demo is public by design.

`./run_qa.sh` needs about 30 GB of GPU memory and, because each persistent
loader worker caches the decoded training images (`--cache_images`), host
RAM of roughly 3.6 GiB per worker for COCO (~72 GiB at the default
`WORKERS=20`, the shipped runs' setting; VOC is about 0.5 GiB per
worker) — set `WORKERS=4` on a smaller host. `FAITHFUL=1 ./run_qa.sh`
trains the same effective-batch-64 recipe as 8 × 8 gradient accumulation
on a small GPU, without the cache (see Speed for the small differences
between the two forms). First use of `qa_calibrate.py` or
`qa_ablation.py` downloads the two torchvision COCO detectors (~92 MB)
that generate detector candidates; training additionally downloads the
ImageNet ResNet-50 weights (~103 MB). Environment knobs of `run_qa.sh`:
`QA_DATASET` (`coco` or `voc`, required), `OUT` (output directory,
default `artifacts_$QA_DATASET`), `WORKERS` (loader workers, default 20;
not passed with `FAITHFUL`, which runs the trainer's default of 4;
throughput only — candidate draws are deterministic per sample),
`RETRIES` (training attempts including the first, default 20; each retry
resumes from `$OUT/checkpoint.pth` after 60 s; configuration errors are
not retried), `FAITHFUL` (any non-empty value);
`qa_interactive_demo.py` serves http://127.0.0.1:7865 (`--host`,
`--port`).

## Weights

The two released decision models are not stored in the repository (about
155 MB each) but hosted in Azure Blob Storage. Download them into the paths
the commands above expect, and verify them:

```bash
curl -fL -o artifacts_voc/best_ema_calibrated.pth  https://labelqa.blob.core.windows.net/checkpoints/v1.0/unified__artifacts_voc__best_ema_calibrated.pth
curl -fL -o artifacts_coco/best_ema_calibrated.pth https://labelqa.blob.core.windows.net/checkpoints/v1.0/unified__artifacts_coco__best_ema_calibrated.pth
sha256sum artifacts_voc/best_ema_calibrated.pth artifacts_coco/best_ema_calibrated.pth
```

| file | SHA-256 | bytes |
| --- | --- | --- |
| `artifacts_voc/best_ema_calibrated.pth` | `738531d0923755983ac7fb697c48f5d40ef23083b3dc6c37f3f78ef4b616b757` | 154517793 |
| `artifacts_coco/best_ema_calibrated.pth` | `91f3aed5e19a452c4008cc90a0d93a90657065938f9f17d64f65729172ef546a` | 154738209 |

Without them, `./run_qa.sh` retrains each model from scratch (60
epochs), and `qa_calibrate.py` then writes its `best_ema_calibrated.pth`
(see [Run](#run)).

## Results

All numbers below come from one run of `run_qa.sh` per dataset
(60 epochs, seed 0, ResNet-50 image trunk, per-box head), the
calibration step, and the protocol, ablation and demo commands of
[Run](#run), and each is backed by a file in `artifacts_<dataset>/`.

**Training and validation** (`artifacts_<dataset>/train.out`,
`training_log.json`, `calibrate.out`). The best validation ROC-AUC of the
raw and the EMA weights over the 60 epochs, and the calibrated model's
validation accuracy:

| dataset | validation ROC-AUC (raw / EMA) | validation accuracy at the validation-tuned threshold (in-sample) | record |
| --- | --- | --- | --- |
| VOC | **0.940 / 0.940** | 0.886 | `training_log.json`, `calibrate.out` |
| COCO | **0.930 / 0.930** | 0.863 | `training_log.json`, `calibrate.out` |

The released EMA checkpoint is epoch 34 on VOC and epoch 36 on COCO
(`training_log.json`). The threshold is chosen on the same validation
candidates it is reported on; the held-out figure at that threshold is the
protocol pool accuracy, 0.857 on the VOC large set
(`artifacts_voc/protocol.out`).

**Held-out protocol** (`qa_protocol.py`, VOC configuration; record in
`artifacts_voc/protocol.json` / `protocol.out`). This applies the *paper's*
A1–A3/B1–B2 error taxonomy to the VOC large set (5 717 held-out images)
under the thesis band metric with `p_exact 0.25`, so its numbers are not
comparable with the thesis's Table 8.5 or with any thesis-generator
evaluation; they measure how the released VOC model handles the paper's
five error types, alone and in every combination. Pool of one candidate
per held-out image — good, or corrupted by one uniformly chosen single
error type, with equal probability, every candidate labelled by the band
metric (5 717 candidates: 2 884 good, 2 833 bad) — at P(good) ≥ 0.5:
accuracy **0.857**, ROC-AUC **0.916** (bad class P/R/F1
0.926 / 0.773 / 0.843, good class 0.808 / 0.939 / 0.869). Mean accuracy
over the 31 error combinations (3 draws of 800 candidates each):
**0.942**; singletons A1 0.873, A2 0.547, A3 0.938, B1 0.925, B2 0.965.
Every combination containing a B-type error scores 0.93–0.97; A2 alone —
offsets of 0.5–5% of the box dimension, many of them below one letterbox
pixel at 224 px and so invisible in the rendered planes — is the one
combination at chance (mean over the other 30 combinations 0.955). The
COCO configuration was not run under this protocol.

**Input ablation** (`qa_ablation.py`, records in
`artifacts_<dataset>/ablation.out`): the same candidates scored three
times — with the true image, with a mismatched image (another sample's)
and with a blank (zeroed) image. The validation population holds one
good-requested and one bad-requested candidate per image, adjacent (every
label is re-derived by the metric, so an in-band detector candidate in the
bad slot counts as good), so the mismatched variant rolls the batch by
two; rolling by one would hand every negative its own image back and
corrupt only the positives. ROC-AUC per condition:

| dataset | true image | mismatched image | blank image |
| --- | --- | --- | --- |
| VOC | 0.940 | 0.568 | 0.578 |
| COCO | 0.930 | 0.543 | 0.553 |

Two things follow. A wrong image costs almost everything: of the
discriminative margin over chance (ROC-AUC − 0.5), only 15% survives on
VOC and 10% on COCO, so the verdict is carried by the image and the label
together, not by the label alone. And a wrong image is rejected outright:
under a mismatched image the model scores essentially every candidate as
bad — mean P(good) 0.023 for positives and 0.017 for negatives on VOC
(0.012 / 0.016 on COCO), below even the blank-image response of
0.086 / 0.061 (0.073 / 0.069) — so the ~0.55 ROC-AUC is uniform
rejection rather than indifference, the response the image-swap
negatives (`--swap_share 0.1`) train for. With a blank or mismatched
image the models fall to 0.54–0.58; what the mismatched row shows is
that an unsupported label is treated as bad, not that the model ranks
candidates without the image.

**Five-sample demo — 5/5** (COCO configuration, record in
`artifacts_coco/demo.out`): 0.757 / 0.042 / 0.963 / 0.966 / 0.045
against targets Good/Bad/Good/Good/Bad. The `+35%` column re-scores
each sample with every box expanded by 35% — each edge moves 17.5% of the
box dimension, 3.5 × the 0.05 band, so the band metric calls every
expanded label Bad; the model scores 0.318 / 0.020 / 0.884 / 0.928 /
0.152, calling 3 of the 5 Bad and keeping samples 3 and 4 Good.

**Caveats.** A single seed per dataset; the validation numbers are on the
set that selected the checkpoint and tuned the threshold; only the VOC
configuration has a held-out protocol record.

## Data

**COCO**: images and `instances_*.json` from
[cocodataset.org](https://cocodataset.org). The model trains on the
val2017 split (as the paper does), 5% of it (227 images) held out for
validation and calibration; train2017 is the paper's held-out test pool,
from which the five demo photographs come (the COCO configuration was
not run under the held-out protocol). **VOC**:
`download_pascal_voc_2012_dataset.py` fetches VOC 2012;
`prepare_voc_data.py` packs the per-image annotations; the validation
split trains (the thesis's small set; 20% of it, 1 164 images, is held
out for validation and calibration), the train split is the held-out
evaluation pool. The five demo photographs are fetched once
by `python3 samples/fetch_samples.py` (~1 MB; not redistributed — see
`NOTICE` and `samples/ATTRIBUTION.md`).

## License

AGPL-3.0-only (`LICENSE`) for the source code and the two released
checkpoints `artifacts_voc/best_ema_calibrated.pth` and
`artifacts_coco/best_ema_calibrated.pth` alike; `NOTICE` carries the
third-party attribution for the ImageNet-derived tensors they embed
(torchvision, BSD-3) and the dataset material.
