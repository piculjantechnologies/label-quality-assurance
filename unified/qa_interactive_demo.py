"""Interactive bounding-box editor for the unified pipeline.

Serves the selected dataset's pool with a unified checkpoint: move,
resize, delete, relabel and add boxes on a canvas; every edit reports the
model's P(good) and the band metric's verdict side by side, with the
uncertainty bands drawn on the image. Boxes are edited in original image
coordinates; the model input is rendered server-side via
qa_data.get_sample exactly as in training. The bands are drawn from the
original-pixel boxes while the verdict is computed on the boxes as
rendered (integer letterbox pixels), so at the band edge the drawing and
the verdict can disagree by up to one letterbox pixel.

Serves whichever datasets are configured; when both are, the page's
dataset selector switches between them live (each keeps its own model,
pool and class list).

"upload photo" (or dropping an image on the page) scores your own photo
with the selected dataset's model: annotate it with that dataset's
classes and the model's P(good) follows every edit. A photo has no
ground truth, so the metric has nothing to compare against and only the
model's verdict is shown. A photo larger than the dataset's own image
scale (longest side 640 px for COCO, 500 px for VOC) is first
downscaled to it. `--uploads-only` skips the pools and serves every
dataset whose checkpoint is present, for photos only.

Every pool image is shown with its provenance: a COCO image with the
Flickr source and licence from the annotation file, a VOC image with the
VOC 2012 source and terms of use. --permissive-licences keeps only the
COCO photographs under CC BY, CC BY-SA, no known restrictions or US
Government work (the VOC pool is not filtered).

Hardened for public exposure: body cap 12 MB, header-only image size check
before decoding, one inference at a time, 32 connections, 30 s socket
timeout, generic error text, access log; run behind a reverse proxy that
terminates TLS and rate-limits (see the README).

    VOC_POOL=~/fiftyone/voc-2012/train \\
    COCO_IMAGES=... COCO_ANNOTATIONS=... \\
        python3 qa_interactive_demo.py --heldout coco   # http://127.0.0.1:7865
    python3 qa_interactive_demo.py --uploads-only   # your own photos only
    python3 qa_interactive_demo.py --heldout coco --permissive-licences

`--heldout <dataset>` restricts that dataset's pool to the training
run's held-out validation slice (use it when the pool is the training
source, as COCO's val2017 is).
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

BACKENDS = {}       # name -> {ds, qa_data, model, classes, samples, pool, credits}
DEFAULT = None
UPLOAD_SIDE = {"coco": 640, "voc": 500}   # each dataset's own image scale
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
VOC_CREDIT = {"source": "Pascal VOC 2012",
              "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/",
              "licence": "Flickr photograph, VOC terms of use",
              "licence_url": "http://host.robots.ox.ac.uk/pascal/VOC/"}
UPLOADS = OrderedDict()
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


def dataset_of(value):
    """A served dataset's name from a query string or a JSON field (the
    default dataset when absent), else ClientError."""
    if value is None or value == "":
        return DEFAULT
    if not isinstance(value, str) or value not in BACKENDS:
        raise ClientError(f"unknown dataset {str(value)[:40]!r}")
    return value


def load_backend(name):
    """Import an independent copy of dataset/qa_data/qa_model for one
    dataset. The modules bind their class list and resolution at import
    time, so each dataset needs its own module instances; swapping them
    out of sys.modules around the import keeps the two sets apart."""
    os.environ["QA_DATASET"] = name
    saved = {k: sys.modules.pop(k, None)
             for k in ("dataset", "dataset_coco", "dataset_voc",
                       "qa_data", "qa_model")}
    try:
        import dataset as ds
        import qa_data as qd
        import qa_model as qm
        mods = (ds, qd, qm)
    finally:
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v
    return mods


def backend(name=None):
    return BACKENDS[name or DEFAULT]


# ------------------------------------------------------------------ data --
@functools.lru_cache(maxsize=16)
def _load_sample(name, index):
    b = backend(name)
    path, ann, w, h = b["samples"][index]
    fname = os.path.basename(path)
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"image not found: {fname}")
    return {"name": os.path.splitext(fname)[0], "fname": fname,
            "image": image, "w": w, "h": h, "ann": ann}


def load_sample(index, name=None):
    """Samples are immutable; cache the recent ones (multi-tab browsing)."""
    b = backend(name)
    return _load_sample(name or DEFAULT, index % len(b["samples"]))


def coco_credits(ann_path):
    """file_name -> (flickr_url, licence name, licence url, licence id).

    Read from the instances file the COCO pool is built from: COCO
    records each photograph's Flickr source and one of eight licence ids.
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


def credit_for(sample, name):
    """Provenance of a pool image for the page's caption; None for a photo."""
    if sample.get("upload"):
        return None
    if name == "voc":
        return dict(VOC_CREDIT)
    url, licence, licence_url, _ = backend(name)["credits"].get(
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
    now = time.monotonic()
    for old in [k for k, s in UPLOADS.items() if now - s["ts"] > UPLOAD_TTL]:
        del UPLOADS[old]


def sweep_uploads():
    """Background sweep, so an idle server still forgets expired photos."""
    while True:
        time.sleep(60)
        with UPLOAD_LOCK:
            purge_expired_uploads()


def add_upload(req, name):
    """Decode an uploaded photo into a sample with no ground truth, for
    dataset `name`'s model.

    OpenCV applies the EXIF orientation while decoding, and the decoded
    image is what the page displays, so the pixels being annotated are
    the pixels the model scores. A photo larger than the dataset's own
    image scale is area-downscaled to it, which keeps the editor's
    original-coordinate canvas at the size of the training images; the
    model then letterboxes it via qa_data.get_sample as usual.

    Photos live only in this process's memory: at most UPLOAD_KEEP of
    them, each for UPLOAD_TTL seconds since it was last scored; nothing
    is written to disk or logged.
    """
    image = decode_upload(req)
    orig_h, orig_w = image.shape[:2]
    s = UPLOAD_SIDE.get(name, backend(name)["qa_data"].RES) / \
        max(orig_h, orig_w)
    if s < 1:
        image = cv2.resize(image, (max(1, round(orig_w * s)),
                                   max(1, round(orig_h * s))),
                           interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    uid = secrets.token_urlsafe(16)
    stem = os.path.splitext(os.path.basename(str(req.get("name", ""))))[0]
    sample = {"name": stem[:80] or "photo", "image": image, "w": w, "h": h,
              "ann": {}, "upload": uid, "dataset": name,
              "original": [orig_w, orig_h], "ts": time.monotonic()}
    with UPLOAD_LOCK:
        purge_expired_uploads()
        UPLOADS[uid] = sample
        while len(UPLOADS) > UPLOAD_KEEP:
            UPLOADS.popitem(last=False)
    return sample


def sample_for(req, name):
    """The dataset sample or the uploaded photo a request refers to."""
    uid = req.get("upload")
    if uid:
        if not isinstance(uid, str):
            raise ClientError("bad upload id")
        with UPLOAD_LOCK:
            purge_expired_uploads()
            sample = UPLOADS.get(uid)
            if sample is not None:        # scoring keeps a photo alive
                UPLOADS.move_to_end(uid)
                sample["ts"] = time.monotonic()
        if sample is None:
            raise ClientError("this photo is no longer held by the server "
                              "— it is re-uploaded automatically",
                              code="upload_expired")
        return sample
    if not backend(name)["samples"]:
        raise ClientError(f"no {name} pool loaded (--uploads-only)")
    return load_sample(to_index(req.get("index", 0)), name)


def sample_payload(sample, name):
    ok, jpg = cv2.imencode(".jpg", sample["image"],
                           [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"dataset": name, "name": sample["name"],
            "w": sample["w"], "h": sample["h"],
            "image": base64.b64encode(jpg.tobytes()).decode(),
            "image_mime": "image/jpeg",
            "boxes": boxes_payload(sample, name),
            "regions": regions_payload(sample, name),
            "classes": backend(name)["classes"],
            "credit": credit_for(sample, name)}


@functools.lru_cache(maxsize=256)
def _sample_json(name, index):
    """The encoded GET /api/sample payload of one pool index (immutable)."""
    b = backend(name)
    sample = load_sample(index, name)
    return json.dumps({**sample_payload(sample, name), "index": index,
                       "total": len(b["samples"]),
                       "pool": b["pool"]}).encode()


def regions_payload(sample, name=None):
    b = backend(name)
    qa_data, CLASSES = b["qa_data"], b["classes"]
    out = []
    a, bta = qa_data.ALPHA, qa_data.BETA
    img_w, img_h = sample["w"], sample["h"]
    for cls, boxes in sorted(sample["ann"].items()):
        for x, y, bw, bh in boxes:
            out.append({
                "cls": CLASSES[cls],
                "outer": [max(0.0, x - a * bw), max(0.0, y - a * bh),
                          min(float(img_w), x + bw + a * bw),
                          min(float(img_h), y + bh + a * bh)],
                "inner": [x + bta * bw, y + bta * bh,
                          x + bw - bta * bw, y + bh - bta * bh],
                "gt": [x, y, x + bw, y + bh]})
    return out


def boxes_payload(sample, name=None):
    CLASSES = backend(name)["classes"]
    return [{"cls": CLASSES[cls], "box": [x, y, x + w, y + h]}
            for cls, boxes in sorted(sample["ann"].items())
            for x, y, w, h in boxes]


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_boxes(items, name=None):
    """Validate a client's box list -> [(class name, [x1, y1, x2, y2])].

    At most MAX_BOXES entries; every class a known name of dataset
    `name`; every box four finite numbers, clipped to +/-COORD_LIMIT.
    Anything else is a ClientError, so no caller ever rasterises an
    unchecked box.
    """
    NAME2IDX = backend(name)["name2idx"]
    if not isinstance(items, list):
        raise ClientError("boxes must be a list")
    if len(items) > MAX_BOXES:
        raise ClientError(f"too many boxes (limit {MAX_BOXES})")
    out = []
    for item in items:
        if not isinstance(item, dict):
            raise ClientError("bad box")
        cls_name = item.get("cls")
        if not isinstance(cls_name, str) or cls_name not in NAME2IDX:
            raise ClientError(f"unknown class {str(cls_name)[:40]!r}")
        box = item.get("box")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or not all(_is_number(c) and math.isfinite(c) for c in box)):
            raise ClientError("bad box")
        out.append((cls_name, [min(COORD_LIMIT, max(-COORD_LIMIT, float(c)))
                               for c in box]))
    return out


def to_bbs(items, name=None):
    """[{cls, box:[x1,y1,x2,y2]}] -> {class_index: [[x1,y1,x2,y2], ...]}"""
    NAME2IDX = backend(name)["name2idx"]
    bbs = {}
    for cls_name, (x1, y1, x2, y2) in check_boxes(items, name):
        if x2 > x1 and y2 > y1:
            bbs.setdefault(NAME2IDX[cls_name], []).append([x1, y1, x2, y2])
    return bbs


# ---------------------------------------------------------------- metric --
def metric_verdict(bbs, sample, name=None):
    b = backend(name)
    qa_data, CLASSES = b["qa_data"], b["classes"]
    old_w, old_h = sample["w"], sample["h"]
    resized_dim = qa_data.RES
    scale = resized_dim / max(old_w, old_h)
    new_w = max(1, int(old_w * scale))
    new_h = max(1, int(old_h * scale))
    gt_rs = qa_data.boxes_from_packed(sample["ann"], old_w, old_h,
                                      new_w, new_h)
    regions = qa_data.build_regions(gt_rs, new_w, new_h)
    # Same boxes_from_packed transform as the GT (clip + MIN_BOX_PX), then
    # int-truncate exactly as build_regions truncates band endpoints, so a
    # verbatim ground truth can never leave its own band.
    packed = {int(cls): [[x1, y1, x2 - x1, y2 - y1]
                         for x1, y1, x2, y2 in boxes]
              for cls, boxes in bbs.items()}
    cand = {c: [[int(v) for v in b] for b in boxes]
            for c, boxes in qa_data.boxes_from_packed(
                packed, old_w, old_h, new_w, new_h).items()}
    bad = qa_data.is_label_negative(cand, regions)
    issues = []
    for c in range(qa_data.NUM_CLASSES):
        n_c, n_r = len(cand.get(c, [])), len(regions[c])
        if n_c != n_r:
            issues.append(f"{CLASSES[c]}: {n_c} boxes for {n_r} objects")
    if bad and not issues:
        issues.append("a box lies outside its uncertainty band")
    return (not bad), issues


# ----------------------------------------------------------------- model --
class inference_slot:
    """One decode/forward at a time; Busy when the slot is not free in time."""

    def __enter__(self):
        if not INFER.acquire(timeout=INFER_TIMEOUT):
            raise Busy()

    def __exit__(self, *exc):
        INFER.release()


def predict(sample, bbs, name=None):
    b = backend(name)
    qa_data, model, CLASSES = b["qa_data"], b["model"], b["classes"]
    device = b["device"]
    named = {}
    for cls, boxes in bbs.items():
        named.setdefault(CLASSES[int(cls)], []).extend(
            [list(bx) for bx in boxes])
    img, planes = qa_data.get_sample(sample["image"], named)
    roi = None
    if getattr(model, "per_box", False):
        rows = qa_data.box_rois(named, sample["w"], sample["h"])
        roi = torch.cat([torch.zeros(rows.shape[0], 1),
                         torch.from_numpy(rows)], 1).to(device)
    with torch.no_grad():
        logits = model(
            torch.from_numpy(planes).float().unsqueeze(0).to(device),
            torch.from_numpy(img).float().unsqueeze(0).to(device),
            boxes=roi)
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
        if path == "/api/datasets":
            return self._send({"datasets": sorted(BACKENDS),
                               "default": DEFAULT})
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
        q = parse_qs(urlparse(self.path).query)
        name = dataset_of(q.get("dataset", [None])[0])
        b = backend(name)
        total = len(b["samples"])
        if not total:
            return self._send({"uploads_only": True, "dataset": name,
                               "classes": b["classes"]})
        if q.get("random", ["0"])[0] == "1":
            index = random.randrange(total)
        else:
            index = to_index(q.get("i", ["0"])[0]) % total
        self._send(_sample_json(name, index))

    def _upload(self):
        req = self._read_body()
        name = dataset_of(req.get("dataset"))
        with inference_slot():
            sample = add_upload(req, name)
        self._send({**sample_payload(sample, name),
                    "upload": sample["upload"],
                    "original": sample["original"], "pool": "your photo"})

    def _predict(self):
        req = self._read_body()
        name = dataset_of(req.get("dataset"))
        items = req.get("boxes", [])
        if not isinstance(items, list):        # cheap checks first
            raise ClientError("boxes must be a list")
        if len(items) > MAX_BOXES:
            raise ClientError(f"too many boxes (limit {MAX_BOXES})")
        sample = sample_for(req, name)
        name = sample.get("dataset", name)      # a photo keeps its model
        bbs = to_bbs(items, name)               # checked against its classes
        with inference_slot():
            p_good, p_bad = predict(sample, bbs, name)
        verdict = {"p_good": p_good, "p_bad": p_bad,
                   "model_says": "good" if p_good >= 0.5 else "bad"}
        if sample.get("upload"):     # no ground truth, so no metric
            return self._send({**verdict, "metric_says": None,
                               "agree": None, "issues": []})
        metric_good, issues = metric_verdict(bbs, sample, name)
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
    ap.add_argument("--datasets", default="",
                    help="comma list (coco,voc); default: every dataset "
                         "whose data environment variables are set")
    ap.add_argument("--coco_checkpoint", default="")
    ap.add_argument("--voc_checkpoint", default="")
    ap.add_argument("--heldout", default="",
                    help="comma list of datasets to restrict to the "
                         "training run's held-out validation slice")
    ap.add_argument("--uploads-only", action="store_true",
                    help="skip the pools and score only photos uploaded "
                         "in the page (default: every dataset whose "
                         "checkpoint is present)")
    ap.add_argument("--permissive-licences", action="store_true",
                    help="keep only COCO pool images whose licence is "
                         "CC BY, CC BY-SA, no known restrictions or US "
                         "Government work (licence ids 4, 5, 7, 8); the "
                         "VOC pool is not filtered")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7865)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    def checkpoint(name):
        return getattr(args, f"{name}_checkpoint", "") or os.path.join(
            _HERE, f"artifacts_{name}", "best_ema_calibrated.pth")

    if args.uploads_only:
        available = [n for n in ("coco", "voc")
                     if os.path.isfile(checkpoint(n))]
    else:
        available = [n for n, ready in
                     (("coco", bool(os.environ.get("COCO_ANNOTATIONS")) and
                       os.path.isdir(os.path.expanduser(
                           os.environ.get("COCO_IMAGES", "")))),
                      ("voc", bool(os.environ.get("VOC_POOL")))) if ready]
    wanted = [n.strip() for n in args.datasets.split(",") if n.strip()] \
        or available
    if not wanted:
        sys.exit("no checkpoint found in artifacts_coco/ or artifacts_voc/"
                 if args.uploads_only else
                 "set COCO_IMAGES/COCO_ANNOTATIONS and/or VOC_POOL — or "
                 "pass --uploads-only to score your own photos")
    heldout = {n.strip() for n in args.heldout.split(",") if n.strip()}

    global DEFAULT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in wanted:
        ckpt = checkpoint(name)
        if not os.path.isfile(ckpt):
            sys.exit(f"checkpoint not found for {name}: {ckpt}")
        ds, qd, qm = load_backend(name)
        credits = {}
        if args.uploads_only:
            samples, pool = [], None
        else:
            src = getattr(ds, "POOL_DIR", "") or getattr(ds, "IMAGES",
                                                         "pool")
            print(f"[{name}] loading pool from {src} ...", flush=True)
            samples = ds.load_pool()
            if not samples:
                sys.exit(f"empty pool at {src}")
            if name in heldout:
                rng = random.Random(0)
                rng.shuffle(samples)
                samples = samples[:max(10,
                                       int(len(samples) * ds.VAL_SPLIT))]
            pool = os.path.basename(str(src).rstrip("/")) + \
                (" (held-out slice)" if name in heldout else "")
            if name == "coco":
                # provenance (Flickr source, licence) per pool image,
                # from the instances file the pool was just built from
                credits = coco_credits(ds.ANNOTATIONS)
                if args.permissive_licences:
                    before = len(samples)
                    samples = [
                        s for s in samples
                        if credits.get(os.path.basename(s[0]),
                                       (None,) * 4)[3]
                        in PERMISSIVE_LICENCES]
                    print(f"[coco] --permissive-licences: {len(samples)} "
                          f"of {before} images kept (CC BY, CC BY-SA, no "
                          f"known restrictions, US Government work)",
                          flush=True)
                    if not samples:
                        sys.exit("no COCO pool images left under "
                                 "--permissive-licences")
        model = qm.load_any(ckpt, device)
        model.eval()
        BACKENDS[name] = {
            "ds": ds, "qa_data": qd, "model": model, "device": device,
            "classes": ds.CLASSES,
            "name2idx": {c: i for i, c in enumerate(ds.CLASSES)},
            "samples": samples, "pool": pool, "credits": credits}
        print(f"[{name}] "
              + ("uploads only" if args.uploads_only
                 else f"{len(samples)} samples")
              + f", {len(ds.CLASSES)} classes, model {ckpt} on {device}",
              flush=True)
    DEFAULT = wanted[0]
    print(f"\n  ->  http://{args.host}:{args.port}"
          f"   (datasets: {', '.join(wanted)})\n", flush=True)
    threading.Thread(target=sweep_uploads, daemon=True).start()
    Server((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
