"""Interactive bounding-box editor over held-out Pascal VOC samples.

Loads an image from the large set (the VOC train split, which training
never uses; --split validation browses the training pool instead), lets
you move, resize, delete, relabel and add boxes on a canvas, and after
every edit reports two verdicts side by side: the model's P(good), and
the uncertainty-region metric (Equation 7.1, two-sided bands,
alpha = beta = 0.05). The bands are drawn on the image, so the tolerance
each box has before the metric flips is visible directly.

"upload photo" (or dropping an image on the page) scores your own photo
instead: annotate it with the 20 VOC classes and the model's P(good)
follows every edit. A photo has no ground truth, so the metric has
nothing to compare against and only the model's verdict is shown.

The editor works in the model's resized coordinate space (longest side
224 px, zoomed for editing) — the space the thesis pipeline itself uses.
An uploaded photo takes the same path, after first being downscaled to
VOC's scale (longest side 500 px) if it is larger.

Every pool image is shown with its provenance (Pascal VOC 2012: Flickr
photographs under the VOC terms of use).

Hardened for public exposure: body cap 12 MB, header-only image size check
before decoding, one inference at a time, 32 connections, 30 s socket
timeout, generic error text, access log; run behind a reverse proxy that
terminates TLS and rate-limits (see the README).

Run (the pool comes from prepare_voc_data.py, see the README):
    python3 interactive_demo.py                 # http://127.0.0.1:7863
    python3 interactive_demo.py --split validation --port 8080
    python3 interactive_demo.py --uploads-only  # your own photos, no VOC
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

from data_loader import (MAPPING, NUM_CLASSES, RES, VOC_CLASSES,
                         _has_perfect_matching, _in_band, ann_to_img,
                         build_regions, is_label_negative, normalize_image,
                         render_planes, resize_keep_aspect)
from neural_network import load_model

UPLOAD_SIDE = 500            # VOC's own scale: longest side 500 px
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
# every pool image is a Flickr photograph distributed under the VOC terms
VOC_CREDIT = {"source": "Pascal VOC 2012",
              "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/",
              "licence": "Flickr photograph, VOC terms of use",
              "licence_url": "http://host.robots.ox.ac.uk/pascal/VOC/"}
STATE = {"uploads": OrderedDict()}
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
def list_samples(pool):
    pool_dir = os.path.expanduser(pool)
    files = sorted(os.path.join(pool_dir, f) for f in os.listdir(pool_dir)
                   if f.endswith(".npy"))
    return [f for f in files
            if any(len(v)
                   for v in np.load(f, allow_pickle=True).item().values())]


@functools.lru_cache(maxsize=8)
def _load_sample(index):
    path = STATE["files"][index]
    image = cv2.imread(ann_to_img(path))
    if image is None:
        raise FileNotFoundError(
            f"image not found for {os.path.basename(path)}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    old_h, old_w = image.shape[:2]
    image = resize_keep_aspect(image)
    new_h, new_w = image.shape[:2]
    packed = np.load(path, allow_pickle=True).item()
    boxes = {key: [] for key in range(NUM_CLASSES)}
    for key, segs in packed.items():
        if int(key) >= NUM_CLASSES:
            continue
        for x, y, w, h in segs:
            boxes[int(key)].append(
                [np.clip(x / old_w * new_w, 0, new_w - 1),
                 np.clip(y / old_h * new_h, 0, new_h - 1),
                 np.clip((x + w) / old_w * new_w, 0, new_w - 1),
                 np.clip((y + h) / old_h * new_h, 0, new_h - 1)])
    regions = build_regions(boxes, new_w, new_h)
    return {"name": os.path.basename(path).split(".")[0],
            "image": image, "w": new_w, "h": new_h, "regions": regions}


def load_sample(index):
    """Samples are immutable; cache the recent ones (multi-tab browsing)."""
    return _load_sample(index % len(STATE["files"]))


def credit_for(sample):
    """Provenance of a pool image for the page's caption; None for a photo."""
    if sample.get("upload"):
        return None
    return dict(VOC_CREDIT)


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
    the pixels the model scores. A photo larger than VOC's scale is
    area-downscaled to it first; from there it takes the pipeline's own
    resize to 224 px, as every training image did.

    Photos live only in this process's memory: at most UPLOAD_KEEP of
    them, each for UPLOAD_TTL seconds since it was last scored; nothing
    is written to disk or logged.
    """
    image = cv2.cvtColor(decode_upload(req), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    s = UPLOAD_SIDE / max(orig_h, orig_w)
    if s < 1:
        image = cv2.resize(image, (max(1, round(orig_w * s)),
                                   max(1, round(orig_h * s))),
                           interpolation=cv2.INTER_AREA)
    image = resize_keep_aspect(image)
    h, w = image.shape[:2]
    uid = secrets.token_urlsafe(16)
    name = os.path.splitext(os.path.basename(str(req.get("name", ""))))[0]
    sample = {"name": name[:80] or "photo", "image": image, "w": w, "h": h,
              "regions": build_regions({}, w, h), "upload": uid,
              "original": [orig_w, orig_h], "ts": time.monotonic()}
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
    if not STATE["files"]:
        raise ClientError("no VOC pool loaded (--uploads-only)")
    return load_sample(to_index(req.get("index", 0)))


def regions_payload(sample):
    """Uncertainty bands as an outer and an inner rectangle per GT box.

    Each coordinate's band is two-sided (outward alpha, inward beta); a
    candidate box is good exactly when each of its four edges lies
    between the outer and the inner rectangle.
    """
    out = []
    for cls in range(NUM_CLASSES):
        for r in sample["regions"][cls]:
            out.append({"cls": VOC_CLASSES[cls],
                        "outer": [r['x1'][0], r['y1'][0],
                                  r['x2'][0], r['y2'][0]],
                        "inner": [r['x1'][1], r['y1'][1],
                                  r['x2'][1], r['y2'][1]],
                        "gt": list(r['gt'])})
    return out


def boxes_payload(sample):
    return [{"cls": VOC_CLASSES[cls], "box": list(r['gt'])}
            for cls in range(NUM_CLASSES) for r in sample["regions"][cls]]


def sample_payload(sample):
    bgr = cv2.cvtColor(sample["image"], cv2.COLOR_RGB2BGR)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"name": sample["name"], "w": sample["w"], "h": sample["h"],
            "image": base64.b64encode(jpg.tobytes()).decode(),
            "image_mime": "image/jpeg",
            "boxes": boxes_payload(sample),
            "regions": regions_payload(sample),
            "classes": VOC_CLASSES,
            "credit": credit_for(sample)}


@functools.lru_cache(maxsize=256)
def _sample_json(index):
    """The encoded GET /api/sample payload of one pool index (immutable)."""
    sample = load_sample(index)
    return json.dumps({**sample_payload(sample), "index": index,
                       "total": len(STATE["files"]),
                       "pool": STATE["pool"]}).encode()


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
        if not isinstance(name, str) or name not in MAPPING:
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
    bbs = {key: [] for key in range(NUM_CLASSES)}
    for name, (x1, y1, x2, y2) in check_boxes(items):
        if x2 > x1 and y2 > y1:
            bbs[MAPPING[name]].append([x1, y1, x2, y2])
    return bbs


def diagnose(bbs, regions):
    """Human-readable reasons the metric rejects a label."""
    issues = []
    for key in range(NUM_CLASSES):
        boxes, regs = bbs.get(key, []), regions[key]
        name = VOC_CLASSES[key]
        if len(boxes) != len(regs):
            issues.append(f"{name}: {len(boxes)} box(es) but {len(regs)} "
                          f"ground-truth region(s)")
            continue
        if not regs:
            continue
        adjacency = [[j for j, b in enumerate(boxes) if _in_band(b, r)]
                     for r in regs]
        unmatched = sum(1 for a in adjacency if not a)
        if unmatched:
            issues.append(f"{name}: {unmatched} region(s) with no box "
                          f"inside the uncertainty band")
        elif not _has_perfect_matching(adjacency, len(boxes)):
            issues.append(f"{name}: boxes cannot be matched one-to-one to "
                          f"regions (duplicates covering the same region)")
    return issues


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
    img = normalize_image(sample["image"])
    planes = render_planes(bbs, sample["h"], sample["w"])
    device = STATE["device"]
    logits = STATE["model"](
        torch.from_numpy(img).float().unsqueeze(0).to(device),
        torch.from_numpy(planes).float().unsqueeze(0).to(device))
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
        if not STATE["files"]:
            return self._send({"uploads_only": True, "classes": VOC_CLASSES})
        q = parse_qs(urlparse(self.path).query)
        if q.get("random", ["0"])[0] == "1":
            index = random.randrange(len(STATE["files"]))
        else:
            index = to_index(q.get("i", ["0"])[0]) % len(STATE["files"])
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
        metric_bad = is_label_negative(bbs, sample["regions"])
        self._send({**verdict,
                    "metric_says": "bad" if metric_bad else "good",
                    "agree": (p_good >= 0.5) == (not metric_bad),
                    "issues": diagnose(bbs, sample["regions"])})


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
                    default=os.path.join(_HERE, "artifacts",
                                         "best_model_refit.pth"))
    ap.add_argument("--split", default="train",
                    choices=["train", "validation"],
                    help="train = the held-out large set (default)")
    ap.add_argument("--pool", default="")
    ap.add_argument("--uploads-only", action="store_true",
                    help="skip VOC and score only photos uploaded in the "
                         "page")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7863)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    pool = args.pool or \
        f"~/fiftyone/voc-2012/{args.split}/processed_annotations"
    if not args.uploads_only and not os.path.exists(os.path.expanduser(pool)):
        raise SystemExit(f"annotation pool not found: {pool}\n"
                         f"run prepare_voc_data.py --split {args.split} "
                         f"first, or pass --uploads-only to score your own "
                         f"photos")
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    STATE["device"] = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")
    STATE["model"] = load_model(args.checkpoint, STATE["device"])
    STATE["files"] = [] if args.uploads_only else list_samples(pool)
    variant = "correlation (image-grounded)" if STATE["model"].corr \
        else "listings-as-published (image-blind)"
    if args.uploads_only:
        STATE["pool"] = None
        print("uploads only: no VOC pool, score your own photos", flush=True)
    else:
        STATE["pool"] = ("held-out large set" if args.split == "train"
                         else "training pool")
        print(f"{len(STATE['files'])} samples from the {STATE['pool']}",
              flush=True)
    print(f"model: {args.checkpoint} — {variant} on {STATE['device']}",
          flush=True)
    print(f"\n  ->  http://{args.host}:{args.port}\n", flush=True)
    threading.Thread(target=sweep_uploads, daemon=True).start()
    Server((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
