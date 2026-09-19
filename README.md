# Machine learning-based label quality assurance for object detection

Given an image and a candidate bounding-box label, a two-branch neural network
decides whether the label is good or bad. A label is good when every
ground-truth box is matched by exactly one candidate box of the right class
lying inside an *uncertainty region* around it; anything else is bad. The
network learns that decision from candidates generated programmatically and
from the output of open-source detectors.

This repository holds three implementations of the method: one per
publication, and one that unifies them.

| directory | implements | dataset | network |
| --- | --- | --- | --- |
| [`thesis/`](thesis/) | the 2025 thesis method, with its documented implementation details | Pascal VOC 2012, 20 classes | two ResNet-18 branches, correlation fusion, per-cell head |
| [`paper/`](paper/) | the 2023 paper method, with its documented implementation details | COCO 2017, 80 classes | two ResNet-18 branches, correlation fusion, per-cell head |
| [`unified/`](unified/) | one dataset-parameterised implementation of both, `QA_DATASET=coco\|voc` | COCO 2017 or Pascal VOC 2012 | ResNet-50 image trunk, ResNet-18 label trunk, per-box head |

`thesis/` and `paper/` share one network design and one training recipe,
each applied to its own publication's task: the dataset, the input
resolution (224 × 224 for VOC, 640 × 640 for COCO, both letterboxed), the
goodness metric, the error generator and the evaluation protocol. The
network runs two ResNet-18 branches to stride 32 — an ImageNet-pretrained
image branch, fine-tuned at 0.1 × the learning rate with its BatchNorm
statistics kept at the pretrained values, and a label branch trained from
scratch on one rendered plane per class — and compares their aligned maps
cell by cell in a correlation fusion. A per-cell (raster) head
(`--raster_head`) works on the stride-8 grid: from the image branch's
stride-8 and stride-16 features next to the label planes it scores every
cell the label draws on, and those cells' scores add to the verdict. The
recipe, which each folder's `run.sh` runs end to end, is AdamW with 2
warmup epochs and cosine decay over 160 epochs, an effective batch of 64
with BatchNorm statistics per group of 8, mixed precision and a weight
EMA; negatives mix the publication's own error generator with
image-swap, image-hard and detector candidates (the detector share is
part of the thesis's method and an addition on the paper side); the
epoch with the highest validation ROC-AUC on a fixed validation set,
drawn as the evaluation protocol draws, is kept, and a logistic refit on the held-out
validation images sets the final decision.

`unified/` is a single code base: the files that define the method
(`qa_data.py`, `qa_model.py`, `qa_train.py`, `qa_calibrate.py`, `run_qa.sh`)
are dataset-agnostic, and `dataset.py` selects the configuration from the
environment — `QA_DATASET=coco` (80 classes, 640-px inputs,
`dataset_coco.py`) or `QA_DATASET=voc` (20 classes, 224-px inputs,
`dataset_voc.py`). Both configurations are trained with the identical recipe
(60 epochs, seed 0) and share the thesis's band goodness metric (two-sided,
α = β = 0.05 of the box dimension, decided by region-to-box matching).

Each folder is self-contained, with its own README, `requirements.txt`,
`LICENSE` and `NOTICE`, and documents, as implementation details, every
point where it departs from its publication. Start with the folder for the
publication you want to reproduce, or with `unified/`, whose README lists
its implementation details with respect to both publications.

## Publications

> Pičuljan, N. and Car, Ž. *Machine Learning-Based Label Quality Assurance for
> Object Detection Projects in Requirements Engineering.* Applied Sciences
> 13(10):6234, 2023. [doi:10.3390/app13106234](https://doi.org/10.3390/app13106234)

> Pičuljan, N. *Machine learning-based method for quality assurance of object
> bounding box labels in images.* PhD thesis, University of Zagreb, Faculty of
> Electrical Engineering and Computing, 2025.
> [urn:nbn:hr:168:865107](https://urn.nsk.hr/urn:nbn:hr:168:865107)

## Getting started

Each folder is self-contained, with its own `requirements.txt`, setup steps
and README. The datasets are downloaded at run time and are not
redistributed here, apart from the five small demo label files in each of
`paper/samples/` and `unified/samples/`, which derive from the COCO 2017
annotations (CC BY 4.0; see those folders' `NOTICE`).

Model weights: all four evaluated checkpoints —
`thesis/artifacts/best_model_refit.pth` (106 MB),
`paper/artifacts/best_model_refit.pth` (107 MB) and
`unified/artifacts_{voc,coco}/best_ema_calibrated.pth` (155 MB each), roughly
523 MB together — are hosted in Azure Blob Storage and must be fetched
before running the evaluations and demos:

```bash
bash scripts/fetch_checkpoints.sh
```

The script downloads from
`https://labelqa.blob.core.windows.net/checkpoints/v1.0` (set
`LABELQA_CHECKPOINT_URL` to use a mirror). Expected paths and SHA-256 digests
are listed in [`scripts/checkpoints.json`](scripts/checkpoints.json); each
blob is named after its path with `/` replaced by `__`
(e.g. `paper__artifacts__best_model_refit.pth`). Training does not need them.

## Deploying the demos publicly

Each folder's box editor (`paper/interactive_demo.py`,
`thesis/interactive_demo.py`, `unified/qa_interactive_demo.py`) is a plain
`http.server` process that is hardened for public exposure but is not meant
to face the internet by itself. It binds to `127.0.0.1`, caps every request
body at 12 MB, checks an uploaded image's dimensions from its header before
decoding it (8 000 px per side, 25 megapixels), runs one inference at a
time, holds at most 32 connections with a 30 s socket timeout, answers
errors with generic text and writes a one-line access log.
Uploaded photos are held only in server memory for re-scoring — never
written to disk or logged — and are discarded 30 minutes after they were
last scored, when 64 more recently used photos are held, or when the
server restarts.
Put a reverse proxy in front that terminates TLS, caps the body,
rate-limits and bounds slow clients, and give each demo its own host or
port:

| demo | local port | pool images |
| --- | --- | --- |
| `thesis/interactive_demo.py` | 7863 | Pascal VOC 2012 |
| `paper/interactive_demo.py` | 7864 | COCO 2017 |
| `unified/qa_interactive_demo.py` | 7865 | COCO 2017 and/or Pascal VOC 2012 |

The pages call the API through relative `api/` URLs, so a demo can also
live under a path prefix as long as its page is served at a trailing-slash
URL (`https://<host>/paper/` with `proxy_pass http://127.0.0.1:7864/`).
The paper and unified demos take `--permissive-licences`, which restricts
the COCO pool to photographs under CC BY, CC BY-SA, no-known-restrictions
or US-Government licences; every pool image is captioned with its source
and licence either way, and `--uploads-only` serves no pool at all.

An nginx server block per demo (the paper demo shown; the two `*_zone`
lines belong in the `http` block):

```nginx
limit_req_zone  $binary_remote_addr zone=labelqa_req:10m rate=10r/s;
limit_conn_zone $binary_remote_addr zone=labelqa_conn:10m;

server {
    listen 443 ssl;
    http2 on;
    server_name paper-demo.example.org;
    ssl_certificate     /etc/letsencrypt/live/paper-demo.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/paper-demo.example.org/privkey.pem;

    client_max_body_size  13m;         # just above the server's own 12 MB cap
    client_body_timeout   30s;
    client_header_timeout 30s;         # nginx's default is 60 s
    limit_req  zone=labelqa_req burst=20 nodelay;
    limit_conn labelqa_conn 10;

    location / {
        proxy_pass         http://127.0.0.1:7864;
        proxy_http_version 1.1;
        proxy_set_header   Host            $host;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 60s;
        proxy_send_timeout 30s;
    }
}
```

Run each demo as its own unprivileged systemd service, with the GPU it may
use pinned and a memory ceiling, so that anything that slips past the caps
kills one demo rather than the host or its neighbours:

```ini
[Unit]
Description=label-QA paper demo
After=network.target

[Service]
User=labelqa
WorkingDirectory=/opt/label-quality-assurance/paper
ExecStart=/opt/label-quality-assurance/.venv/bin/python3 interactive_demo.py \
    --checkpoint artifacts/best_model_refit.pth --host 127.0.0.1 --port 7864 \
    --permissive-licences
Environment=PYTHONUNBUFFERED=1
Environment=CUDA_VISIBLE_DEVICES=0
Environment=COCO_IMAGES=/data/coco/val2017
Environment=COCO_ANNOTATIONS=/data/coco/annotations/instances_val2017.json
MemoryMax=6G
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Before opening a host, probe it from outside with the stdlib-only checker:

```bash
python3 scripts/probe_demo.py https://<host>          # one PASS/FAIL line per check
python3 scripts/probe_demo.py https://<host> --idle   # also the idle-connection timeout (waits 45 s)
```

It exercises the body cap (a negative `Content-Length`), the
decompression-bomb guard (a valid PNG declaring 16 000 × 16 000 px), the box
cap, `NaN` rejection, non-JSON bodies, the generic error text, `HEAD /`,
`GET /?view=api`, the response headers (no server version, `nosniff`,
`X-Frame-Options`, `Cache-Control: no-store` on `api/`) and, with
`--idle`, that idle and half-sent connections are closed by the timeout;
it then uploads a tiny PNG and scores one box to prove the happy path
still works, and exits non-zero if any check fails. Behind a proxy the
idle check sees the proxy's timeouts, not the server's: the block above
bounds idle and half-sent clients at 30 s, and a proxy that allows more
(nginx's `client_header_timeout` defaults to 60 s) needs `--timeout`
raised above its value. Deploy from a committed tree, so that what is
running is exactly a known commit.

## Licence

The source code and every distributed checkpoint — the four listed in
`scripts/checkpoints.json` — are licensed under the GNU Affero
General Public License, version 3 only (AGPL-3.0-only); see
[LICENSE](LICENSE). The checkpoints embed tensors derived from torchvision's
ImageNet-pretrained ResNets; [NOTICE](NOTICE) carries that third-party
attribution (BSD 3-Clause) and the dataset terms, and each folder's own
`NOTICE` states the details for its checkpoint.