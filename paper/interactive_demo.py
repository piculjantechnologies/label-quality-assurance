"""Interactive bounding-box editor over held-out COCO samples.

Loads an image from the run's held-out validation samples (the val2017
split the training run never trains on, selected by the seed and sized
by the val_split read from the training_log.json next to the
checkpoint: 454 images for the shipped run; without that file it falls
back to seed 0 and a 5% split, 227 images; --all browses the full
val2017 pool), lets you move,
resize, delete, relabel and add boxes on a canvas, and after every edit
reports two verdicts side by side: the model's P(good), and the
uncertainty-region metric (the paper's Equations 1-4, box-scaled as in
interpretation 2 of the README). The uncertainty bands are drawn on the
image, so the tolerance each box has before the metric flips is visible
directly.

"upload photo" (or dropping an image on the page) scores your own photo
instead: annotate it with the 80 COCO classes and the model's P(good)
follows every edit. A photo has no ground truth, so the metric has
nothing to compare against and only the model's verdict is shown. A photo
larger than COCO's scale (longest side 640 px) is first downscaled to it.

Boxes are edited in the image's original coordinate space; the model input
(640-px letterbox) is rendered server-side exactly as in training.

Every pool image is shown with its COCO provenance (the Flickr source and
licence from the annotation file); --permissive-licences keeps only the
CC BY, CC BY-SA, no-known-restrictions and US-Government photographs.

Hardened for public exposure: body cap 12 MB, header-only image size check
before decoding, one inference at a time, 32 connections, 30 s socket
timeout, generic error text, access log; run behind a reverse proxy that
terminates TLS and rate-limits (see the README).

Run (needs COCO_IMAGES / COCO_ANNOTATIONS, as for run.sh):
    python3 interactive_demo.py                 # http://127.0.0.1:7864
    python3 interactive_demo.py --port 8080 --all
    python3 interactive_demo.py --uploads-only  # your own photos, no COCO
    python3 interactive_demo.py --permissive-licences
"""

import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "25000000")   # backstop; must precede `import cv2`

import argparse
import base64
import binascii
import functools
import io
import json
import logging
import math
import random
import secrets
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import data_loader as dl
from coco_corrnet import CorrNet

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def load_any(ckpt_path, device=None):
    """Load a CorrNet checkpoint."""
    return CorrNet.from_checkpoint(ckpt_path, device)


NAME2IDX = {n: i for i, n in enumerate(COCO_CLASSES)}
SEED, VAL_SPLIT, EPS = 0, 0.05, 1e-6
UPLOAD_SIDE = 640            # COCO's own scale: longest side 640 px
UPLOAD_BYTES = 12 << 20      # request-body cap (JSON with base64; an 8 MB file inflates to ~11 MB)
MAX_PIXELS = 25_000_000      # decoded-pixel budget checked from the header, before decoding
MAX_SIDE = 8000
MAX_BOXES = 500
UPLOAD_KEEP = 64             # uploaded photos held for re-scoring
UPLOAD_TTL = 30 * 60         # seconds
MAX_CONNECTIONS = 32
INFER_TIMEOUT = 2.0          # seconds to wait for the inference slot before 503
COORD_LIMIT = 1e6            # box coordinates are clipped to +/- this
TOO_LARGE = (f"image too large (limit {MAX_SIDE} px per side, "
             f"{MAX_PIXELS // 1_000_000} megapixels)")
NOT_IMAGE = "could not decode the file as an image (JPEG, PNG, WebP, BMP or TIFF)"
# COCO licence ids 4, 5, 7, 8: CC BY, CC BY-SA, no known restrictions, US Gov
PERMISSIVE_LICENCES = {4, 5, 7, 8}
STATE = {"uploads": OrderedDict(), "credits": {}}
UPLOAD_LOCK = threading.Lock()
INFER = threading.BoundedSemaphore(1)


class ClientError(ValueError):
    """An error whose message may be shown to visitors (HTTP 4xx)."""

    def __init__(self, message, code=None, status=400):
        super().__init__(message)
        self.code = code
        self.status = status


class Busy(Exception):
    """The inference slot was not free within INFER_TIMEOUT."""


def _no_const(name):
    raise ClientError("NaN and Infinity are not allowed")


def to_index(value):
    """A pool index from a query string or a JSON field, else ClientError."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ClientError("bad index")
    try:
        return int(value)
    except ValueError:
        raise ClientError("bad index")


# ------------------------------------------------------------------ data --
@functools.lru_cache(maxsize=8)
def _load_sample(index):
    fname, ann = STATE["samples"][index]
    image = cv2.imread(os.path.join(STATE["images"], fname))
    if image is None:
        raise FileNotFoundError(f"image not found: {os.path.basename(fname)}")
    h, w = image.shape[:2]
    return {"name": os.path.splitext(fname)[0], "fname": fname,
            "image": image, "w": w, "h": h, "ann": ann}


def load_sample(index):
    """Samples are immutable; cache the recent ones (multi-tab browsing)."""
    return _load_sample(index % len(STATE["samples"]))


def coco_credits(ann_path):
    """file_name -> (flickr_url, licence name, licence url, licence id).

    Read from the instances file the pool is built from: COCO records each
    photograph's Flickr source and one of eight licence ids.
    """
    with open(ann_path) as f:
        coco = json.load(f)
    licences = {lic["id"]: (lic.get("name", ""), lic.get("url", ""))
                for lic in coco.get("licenses", [])}
    credits = {}
    for im in coco["images"]:
        name, url = licences.get(im.get("license"), ("", ""))
        credits[im["file_name"]] = (im.get("flickr_url", ""), name, url,
                                    im.get("license"))
    return credits


def credit_for(sample):
    """Provenance of a pool image for the page's caption; None for a photo."""
    if sample.get("upload"):
        return None
    url, licence, licence_url, _ = STATE["credits"].get(
        sample["fname"], ("", "", "", None))
    return {"source": "COCO 2017 val", "url": url,
            "licence": licence, "licence_url": licence_url}


def header_size(raw):
    """(w, h) of an encoded image read from its header; no pixels decoded.

    Rejects, before any decoder touches the pixels, what would not fit
    the MAX_PIXELS / MAX_SIDE budget (decompression bombs included).
    """
    try:
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
    except Image.DecompressionBombError:
        raise ClientError(TOO_LARGE)
    except Exception:
        raise ClientError(NOT_IMAGE)
    if w * h > MAX_PIXELS or max(w, h) > MAX_SIDE:
        raise ClientError(TOO_LARGE)
    return w, h


def decode_upload(req):
    """The uploaded photo's bytes, decoded to BGR after the header check."""
    try:
        raw = base64.b64decode(req.get("image", ""), validate=True)
    except (binascii.Error, TypeError, ValueError):
        raise ClientError("image is not valid base64")
    if not raw:
        raise ClientError("no image data")
    header_size(raw)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ClientError(NOT_IMAGE)
    return image


def purge_expired_uploads():
    """Drop photos not scored for UPLOAD_TTL seconds (caller holds the lock)."""
    uploads = STATE["uploads"]
    now = time.monotonic()
    for old in [k for k, s in uploads.items() if now - s["ts"] > UPLOAD_TTL]:
        del uploads[old]


def sweep_uploads():
    """Background sweep, so an idle server still forgets expired photos."""
    while True:
        time.sleep(60)
        with UPLOAD_LOCK:
            purge_expired_uploads()


def add_upload(req):
    """Decode an uploaded photo into a sample with no ground truth.

    OpenCV applies the EXIF orientation while decoding, and the decoded
    image is what the page displays, so the pixels being annotated are
    the pixels the model scores. A photo larger than COCO's scale is
    area-downscaled to it first, which keeps the model's 640-px letterbox
    close to the training images instead of a hard subsample of a
    multi-megapixel photo.

    Photos live only in this process's memory: at most UPLOAD_KEEP of
    them, each for UPLOAD_TTL seconds since it was last scored; nothing
    is written to disk or logged.
    """
    image = decode_upload(req)
    orig_h, orig_w = image.shape[:2]
    s = UPLOAD_SIDE / max(orig_h, orig_w)
    if s < 1:
        image = cv2.resize(image, (max(1, round(orig_w * s)),
                                   max(1, round(orig_h * s))),
                           interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    uid = secrets.token_urlsafe(16)
    name = os.path.splitext(os.path.basename(str(req.get("name", ""))))[0]
    sample = {"name": name[:80] or "photo", "image": image, "w": w, "h": h,
              "ann": {}, "upload": uid, "original": [orig_w, orig_h],
              "ts": time.monotonic()}
    with UPLOAD_LOCK:
        purge_expired_uploads()
        uploads = STATE["uploads"]
        uploads[uid] = sample
        while len(uploads) > UPLOAD_KEEP:
            uploads.popitem(last=False)
    return sample


def sample_for(req):
    """The dataset sample or the uploaded photo a request refers to."""
    uid = req.get("upload")
    if uid:
        if not isinstance(uid, str):
            raise ClientError("bad upload id")
        with UPLOAD_LOCK:
            purge_expired_uploads()
            sample = STATE["uploads"].get(uid)
            if sample is not None:        # scoring keeps a photo alive
                STATE["uploads"].move_to_end(uid)
                sample["ts"] = time.monotonic()
        if sample is None:
            raise ClientError("this photo is no longer held by the server "
                              "— it is re-uploaded automatically",
                              code="upload_expired")
        return sample
    if not STATE["samples"]:
        raise ClientError("no COCO pool loaded (--uploads-only)")
    return load_sample(to_index(req.get("index", 0)))


def sample_payload(sample):
    ok, jpg = cv2.imencode(".jpg", sample["image"],
                           [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"name": sample["name"], "w": sample["w"], "h": sample["h"],
            "image": base64.b64encode(jpg.tobytes()).decode(),
            "image_mime": "image/jpeg",
            "boxes": boxes_payload(sample),
            "regions": regions_payload(sample),
            "classes": COCO_CLASSES,
            "credit": credit_for(sample)}


@functools.lru_cache(maxsize=256)
def _sample_json(index):
    """The encoded GET /api/sample payload of one pool index (immutable)."""
    sample = load_sample(index)
    return json.dumps({**sample_payload(sample), "index": index,
                       "total": len(STATE["samples"]),
                       "pool": STATE["pool"]}).encode()


def regions_payload(sample):
    """Uncertainty bands as an outer and an inner rectangle per GT box.

    A candidate box is good exactly when each of its four edges lies
    between the outer and the inner rectangle (closed intervals).
    """
    out = []
    for cls, boxes in sorted(sample["ann"].items()):
        for b in boxes:
            (a, b1), (c, d), (e, f), (g, k) = dl._ranges(
                b, sample["w"], sample["h"])
            x, y, bw, bh = b
            out.append({"cls": COCO_CLASSES[cls],
                        "outer": [a, c, f, k],
                        "inner": [b1, d, e, g],
                        "gt": [x, y, x + bw, y + bh]})
    return out


def boxes_payload(sample):
    return [{"cls": COCO_CLASSES[cls],
             "box": [x, y, x + w, y + h]}
            for cls, boxes in sorted(sample["ann"].items())
            for x, y, w, h in boxes]


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_boxes(items):
    """Validate a client's box list -> [(class name, [x1, y1, x2, y2])].

    At most MAX_BOXES entries; every class a known name; every box four
    finite numbers, clipped to +/-COORD_LIMIT. Anything else is a
    ClientError, so no caller ever rasterises an unchecked box.
    """
    if not isinstance(items, list):
        raise ClientError("boxes must be a list")
    if len(items) > MAX_BOXES:
        raise ClientError(f"too many boxes (limit {MAX_BOXES})")
    out = []
    for item in items:
        if not isinstance(item, dict):
            raise ClientError("bad box")
        name = item.get("cls")
        if not isinstance(name, str) or name not in NAME2IDX:
            raise ClientError(f"unknown class {str(name)[:40]!r}")
        box = item.get("box")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or not all(_is_number(c) and math.isfinite(c) for c in box)):
            raise ClientError("bad box")
        out.append((name, [min(COORD_LIMIT, max(-COORD_LIMIT, float(c)))
                           for c in box]))
    return out


def to_bbs(items):
    """[{cls, box:[x1,y1,x2,y2]}] -> {class_index: [[x1,y1,x2,y2], ...]}"""
    bbs = {}
    for name, (x1, y1, x2, y2) in check_boxes(items):
        if x2 > x1 and y2 > y1:
            bbs.setdefault(NAME2IDX[name], []).append([x1, y1, x2, y2])
    return bbs


# ---------------------------------------------------------------- metric --
# The metric below is the whole-label rule of data_loader.label_good
# (README item 12), re-implemented here to report which class breaks it;
# it reads the bands from dl._ranges so the region definition itself has a
# single source. The +/-EPS is a deliberate widening over the exact closed
# intervals, so float round-trips through JSON can never fail a box that
# sits exactly on a band edge.
def _in_band(box, gt_box, img_w, img_h):
    """All four coordinates inside the GT box's uncertainty ranges."""
    ranges = dl._ranges(gt_box, img_w, img_h)
    return all(lo - EPS <= c <= hi + EPS
               for c, (lo, hi) in zip(box, ranges))


def _has_perfect_matching(adjacency, n_boxes):
    """One-to-one region-to-box assignment via augmenting paths."""
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


def metric_verdict(bbs, sample):
    """The paper's rule: every uncertainty region covered by exactly one
    box of its class, no box outside every region, no extra classes."""
    issues = []
    img_w, img_h = sample["w"], sample["h"]
    classes = set(sample["ann"]) | set(bbs)
    for cls in sorted(classes):
        name = COCO_CLASSES[cls]
        gts = sample["ann"].get(cls, [])
        boxes = bbs.get(cls, [])
        if len(boxes) != len(gts):
            issues.append(f"{name}: {len(boxes)} box(es) but {len(gts)} "
                          f"ground-truth region(s)")
            continue
        if not gts:
            continue
        adjacency = [[j for j, b in enumerate(boxes)
                      if _in_band(b, g, img_w, img_h)] for g in gts]
        unmatched = sum(1 for a in adjacency if not a)
        if unmatched:
            issues.append(f"{name}: {unmatched} region(s) with no box "
                          f"inside the uncertainty band")
        elif not _has_perfect_matching(adjacency, len(boxes)):
            issues.append(f"{name}: boxes cannot be matched one-to-one to "
                          f"regions (duplicates covering the same region)")
    return (not issues), issues


# ----------------------------------------------------------------- model --
class inference_slot:
    """One decode/forward at a time; Busy when the slot is not free in time."""

    def __enter__(self):
        if not INFER.acquire(timeout=INFER_TIMEOUT):
            raise Busy()

    def __exit__(self, *exc):
        INFER.release()


@torch.no_grad()
def predict(sample, bbs):
    ann = {cls: [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
           for cls, boxes in bbs.items()}
    cats, background = dl.get_sample(640, sample["image"], ann)
    device = STATE["device"]
    logits = STATE["model"](
        torch.from_numpy(cats).float().unsqueeze(0).to(device),
        torch.from_numpy(background).float().unsqueeze(0).to(device))
    p = torch.softmax(logits.float(), 1)[0]
    return float(p[1]), float(p[0])


# ---------------------------------------------------------------- server --
_CUDA_OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def page():
    with open(os.path.join(_HERE, "interactive_demo.html"), "rb") as f:
        return f.read()


class Handler(BaseHTTPRequestHandler):
    server_version = "labelqa"
    sys_version = ""
    timeout = 30                 # socket timeout; idle peers are dropped

    def log_message(self, fmt, *args):
        logging.info("%s %s", self.client_address[0], fmt % args)

    def version_string(self):
        return self.server_version   # the default appends sys_version + ' '

    def send_error(self, code, message=None, explain=None):
        """http.server's own error replies (bad request line, unsupported
        method or version, oversized headers) as JSON through _send, so
        they carry the same headers as every other answer."""
        self.close_connection = True
        self._send({"error": "not found" if code == 404 else
                    "bad request" if code < 500 else "not supported"},
                   code, extra={"Connection": "close"})

    def _send(self, payload, status=200, ctype="application/json",
              extra=None):
        """Write one response; on HEAD the headers only."""
        body = payload if isinstance(payload, bytes) else \
            json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            if ctype.startswith("application/json"):
                self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _run(self, handler):
        """Run one API handler, mapping its exceptions to responses."""
        try:
            handler()
        except ClientError as e:
            body = {"error": str(e)}
            if e.code:
                body["code"] = e.code
            self._send(body, e.status)
        except Busy:
            self._send({"error": "busy, retry in a moment"}, 503,
                       extra={"Retry-After": "2"})
        except TimeoutError:
            raise                        # handle_one_request drops the peer
        except (_CUDA_OOM, RuntimeError):
            logging.exception("inference failed")
            self._send({"error": "inference unavailable, retry"}, 503,
                       extra={"Retry-After": "2"})
        except Exception:
            logging.exception("request failed")
            self._send({"error": "internal error"}, 500)

    def _read_body(self):
        """The request body as a JSON object, or a ClientError."""
        try:
            n = int(self.headers.get("Content-Length", ""))
        except (TypeError, ValueError):
            raise ClientError("bad request")
        if n < 0 or n > UPLOAD_BYTES:
            raise ClientError(f"request too large (limit "
                              f"{UPLOAD_BYTES >> 20} MB)", status=413)
        body = self.rfile.read(n)
        try:
            req = json.loads(body, parse_constant=_no_const)
        except ClientError:
            raise
        except (ValueError, RecursionError):   # not JSON, bad UTF-8, absurd nesting
            raise ClientError("body is not JSON")
        if not isinstance(req, dict):
            raise ClientError("body must be a JSON object")
        return req

    def do_HEAD(self):
        if urlparse(self.path).path == "/":
            return self._send(page(), ctype="text/html; charset=utf-8")
        self._send({"error": "not found"}, 404)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send(page(), ctype="text/html; charset=utf-8")
        if path == "/api/sample":
            return self._run(self._sample)
        self._send({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/upload":
            return self._run(self._upload)
        if path == "/api/predict":
            return self._run(self._predict)
        self._send({"error": "not found"}, 404)

    def _sample(self):
        if not STATE["samples"]:
            return self._send({"uploads_only": True, "classes": COCO_CLASSES})
        q = parse_qs(urlparse(self.path).query)
        if q.get("random", ["0"])[0] == "1":
            index = random.randrange(len(STATE["samples"]))
        else:
            index = to_index(q.get("i", ["0"])[0]) % len(STATE["samples"])
        self._send(_sample_json(index))

    def _upload(self):
        req = self._read_body()
        with inference_slot():
            sample = add_upload(req)
        self._send({**sample_payload(sample), "upload": sample["upload"],
                    "original": sample["original"], "pool": "your photo"})

    def _predict(self):
        req = self._read_body()
        bbs = to_bbs(req.get("boxes", []))     # cheap checks first
        sample = sample_for(req)
        with inference_slot():
            p_good, p_bad = predict(sample, bbs)
        verdict = {"p_good": p_good, "p_bad": p_bad,
                   "model_says": "good" if p_good >= 0.5 else "bad"}
        if sample.get("upload"):     # no ground truth, so no metric
            return self._send({**verdict, "metric_says": None,
                               "agree": None, "issues": []})
        metric_good, issues = metric_verdict(bbs, sample)
        self._send({**verdict,
                    "metric_says": "good" if metric_good else "bad",
                    "agree": (p_good >= 0.5) == metric_good,
                    "issues": issues})


class Server(ThreadingHTTPServer):
    """ThreadingHTTPServer with at most MAX_CONNECTIONS live connections."""
    daemon_threads = True
    allow_reuse_address = True
    # listen backlog: a burst of connects queues in the kernel instead of
    # being dropped or reset before the connection cap can answer
    request_queue_size = MAX_CONNECTIONS * 4

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\n"
                                b"Retry-After: 2\r\nContent-Length: 0\r\n"
                                b"Connection: close\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def handle_error(self, request, client_address):
        logging.warning("connection error from %s: %s", client_address[0],
                        sys.exc_info()[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",
                    default=os.path.join(_HERE, "artifacts", "best_model_refit.pth"))
    ap.add_argument("--images", default=os.environ.get("COCO_IMAGES", ""))
    ap.add_argument("--annotations",
                    default=os.environ.get("COCO_ANNOTATIONS", ""))
    ap.add_argument("--all", action="store_true",
                    help="browse the full val2017 pool instead of the "
                         "run's held-out validation images")
    ap.add_argument("--pool-name", default="",
                    help="label shown for the image pool in the page "
                         "(default: 'held-out split' or, with --all, "
                         "'full val2017 pool')")
    ap.add_argument("--uploads-only", action="store_true",
                    help="skip COCO and score only photos uploaded in "
                         "the page")
    ap.add_argument("--permissive-licences", action="store_true",
                    help="keep only pool images whose COCO licence is "
                         "CC BY, CC BY-SA, no known restrictions or US "
                         "Government work (licence ids 4, 5, 7, 8)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7864)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.uploads_only:
        if not args.images or not args.annotations:
            sys.exit("set COCO_IMAGES and COCO_ANNOTATIONS (or pass "
                     "--images/--annotations), as for run.sh — or pass "
                     "--uploads-only to score your own photos")
        images = os.path.expanduser(args.images)
        ann_path = os.path.expanduser(args.annotations)
        if not os.path.isdir(images):
            sys.exit(f"COCO_IMAGES is not a directory: {images}")
        if not os.path.isfile(ann_path):
            sys.exit(f"COCO_ANNOTATIONS is not a file: {ann_path}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint} — run ./run.sh "
                 f"first")

    STATE["samples"] = []
    if not args.uploads_only:
        # the held-out slice is defined by the run's seed and val_split,
        # read from the training record when it sits next to the checkpoint
        log = os.path.join(os.path.dirname(args.checkpoint) or ".",
                           "training_log.json")
        if os.path.exists(log):
            a = json.load(open(log)).get("args", {})
            STATE["seed"] = a.get("seed", SEED)
            STATE["val_split"] = a.get("val_split", VAL_SPLIT)
        samples, _ = dl.load_coco(ann_path)
        rng = random.Random(STATE.get("seed", SEED))
        rng.shuffle(samples)
        n_val = max(10, int(len(samples) *
                            STATE.get("val_split", VAL_SPLIT)))
        STATE["samples"] = samples if args.all else samples[:n_val]
        STATE["images"] = images
        STATE["credits"] = coco_credits(ann_path)
        if args.permissive_licences:
            before = len(STATE["samples"])
            STATE["samples"] = [
                s for s in STATE["samples"]
                if STATE["credits"].get(s[0], (None,) * 4)[3]
                in PERMISSIVE_LICENCES]
            print(f"--permissive-licences: {len(STATE['samples'])} of "
                  f"{before} images kept (CC BY, CC BY-SA, no known "
                  f"restrictions, US Government work)", flush=True)
            if not STATE["samples"]:
                sys.exit("no pool images left under --permissive-licences")
    STATE["device"] = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
    STATE["model"] = load_any(args.checkpoint, STATE["device"])
    STATE["model"].eval()
    if args.uploads_only:
        STATE["pool"] = None
        print("uploads only: no COCO pool, score your own photos", flush=True)
    else:
        pool = args.pool_name or ("full val2017 pool" if args.all
                                  else "held-out split")
        STATE["pool"] = pool
        print(f"{len(STATE['samples'])} samples from the {pool}", flush=True)
    print(f"model: {args.checkpoint} on {STATE['device']}", flush=True)
    print(f"\n  ->  http://{args.host}:{args.port}\n", flush=True)
    threading.Thread(target=sweep_uploads, daemon=True).start()
    Server((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
