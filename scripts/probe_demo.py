"""Pre-opening probe for the interactive demo servers.

Sends a demo server the requests a hostile or careless visitor might send
and reports, one PASS/FAIL line per check, whether it answers the way the
hardened servers do. Works alike against the paper server (port 7864),
the thesis server (7863) and the unified server (7865), directly or
behind a reverse proxy that terminates TLS. Standard library only.

    python3 scripts/probe_demo.py http://127.0.0.1:7864
    python3 scripts/probe_demo.py https://demo.example.org --idle
    python3 scripts/probe_demo.py https://demo.example.org/paper/ --idle --timeout 70
    python3 scripts/probe_demo.py https://staging.local:8443 --insecure
    python3 scripts/probe_demo.py http://127.0.0.1:7865 --skip-happy

Checks:
  1  raw POST /api/predict with "Content-Length: -1": 400 or 413, and no
     verdict in the answer
  2  POST /api/upload of a real 16000x16000 PNG (all black, built here
     with zlib, a few hundred KB on the wire): a 4xx answer within 5 s
     whose error text names neither OpenCV nor a file path (the image
     size is read from the header before anything is decoded)
  3  POST /api/upload {}: 400 "no image data"
  4  POST /api/predict with 501 boxes (400 "too many boxes"), with a NaN
     coordinate, with "boxes": "x", with a non-JSON body and with a
     non-numeric Content-Length: 400 each, generic error text
  5  HEAD / 200; GET /?view=api 200 text/html; GET /nope 404
  6  headers: Server names neither Python nor BaseHTTP;
     X-Content-Type-Options: nosniff and X-Frame-Options: DENY on the
     page and on the API; Cache-Control: no-store on /api/*
  7  --idle only: open --connections sockets (default 40), send nothing
     on half of them and a partial request line on the other half, wait
     --timeout seconds (default 45) and require that the server closed
     every one. The demos close an idle socket after 30 s and refuse
     connections beyond 32 with an immediate 503, which counts as closed.
  8  happy path: upload a 4x4 PNG (200 with an upload id), then score one
     box on it (200 with p_good). The unified server is sent the first
     entry of GET /api/datasets as "dataset". --skip-happy skips this
     check when there is no model behind the server.

The predict requests of check 4 carry the id of a tiny upload made first
(or "index": 0 when the upload fails), so a server started with
--uploads-only reaches its box validation instead of answering "no pool".

Exit status is 1 when any check fails, else 0. The URL may carry a path
prefix (https://host/paper/): the demos are served under one at a
trailing-slash URL. --insecure accepts a self-signed certificate. Behind
a reverse proxy the idle check sees the proxy's timeouts, so give
--timeout a value above them (nginx's client_header_timeout is 60 s by
default).
"""

import argparse
import base64
import http.client
import json
import re
import select
import socket
import ssl
import struct
import sys
import time
import zlib
from urllib.parse import urlparse

CONNECT_TIMEOUT = 10.0      # seconds to reach the server
READ_TIMEOUT = 30.0         # seconds to wait for an ordinary answer
BOMB_TIMEOUT = 5.0          # check 2: the header-only size check is quick
BIG_SIDE = 16000            # 256 megapixels, above any sane decode budget
TINY_SIDE = 4
MAX_BOXES = 500             # the servers' cap; check 4 sends one more
# error text that leaks internals: a library name, a traceback, a path
LEAK = re.compile(r"opencv|cv2|traceback|(?:^|[\s(\"'])/\S+/|[a-z]:\\",
                  re.I)

RESULTS = []


class ProbeError(Exception):
    """A check could not get an answer (refused, timed out, malformed)."""


# ------------------------------------------------------------- reporting --
def report(ok, label, detail=""):
    RESULTS.append(ok)
    line = ("PASS  " if ok else "FAIL  ") + label
    print(line + (": " + detail if detail else ""), flush=True)


def note(text):
    print("note  " + text, flush=True)


def leak(text):
    """The first leaking fragment of an error text, or ''."""
    m = LEAK.search(text or "")
    return m.group(0).strip() if m else ""


# ----------------------------------------------------------------- images --
def png(width, height, colour_type, row):
    """A valid 8-bit non-interlaced PNG; row(y) gives one row's samples."""
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    comp = zlib.compressobj(9)
    parts = [comp.compress(b"\x00" + row(y)) for y in range(height)]
    parts.append(comp.flush())
    ihdr = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", b"".join(parts)) + chunk(b"IEND", b""))


def big_png():
    """16000x16000 greyscale, all black: 256 megapixels in ~250 KB."""
    black = bytes(BIG_SIDE)
    return png(BIG_SIDE, BIG_SIDE, 0, lambda y: black)


def tiny_png():
    """4x4 RGB with a gradient, so it is a real (if small) photo."""
    return png(TINY_SIDE, TINY_SIDE, 2,
               lambda y: bytes(v for x in range(TINY_SIDE)
                               for v in (x * 60, y * 60, 128)))


def b64(data):
    return base64.b64encode(data).decode("ascii")


# --------------------------------------------------------------- transport --
class Response:
    def __init__(self, status, headers, body, elapsed):
        self.status, self.headers = status, list(headers)
        self.body, self.elapsed = body, elapsed
        self.json = None
        if "json" in self.header("content-type"):
            try:
                self.json = json.loads(body.decode("utf-8", "replace"))
            except ValueError:
                pass

    def header(self, name):
        for key, value in self.headers:
            if key.lower() == name.lower():
                return value
        return ""

    def has(self, key):
        return isinstance(self.json, dict) and key in self.json

    @property
    def error(self):
        if self.has("error") and isinstance(self.json["error"], str):
            return self.json["error"]
        return ""

    def __str__(self):
        text = str(self.status)
        if self.error:
            text += " " + repr(self.error[:120])
        elif self.json is None and self.body:
            text += " (%s, %d bytes)" % (self.header("content-type")
                                        or "no content-type", len(self.body))
        return text


class Target:
    """One demo server: host, port, TLS context and path prefix."""

    def __init__(self, url, insecure=False):
        u = urlparse(url if "://" in url else "http://" + url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise SystemExit("usage: probe_demo.py http[s]://host[:port][/prefix/]")
        self.tls = u.scheme == "https"
        self.host = u.hostname
        self.port = u.port or (443 if self.tls else 80)
        self.netloc = u.netloc
        self.prefix = u.path.rstrip("/")
        self.ctx = None
        if self.tls:
            self.ctx = ssl.create_default_context()
            if insecure:
                self.ctx.check_hostname = False
                self.ctx.verify_mode = ssl.CERT_NONE

    def path(self, path):
        return self.prefix + path

    def socket(self, timeout):
        s = socket.create_connection((self.host, self.port), timeout)
        if self.tls:
            s = self.ctx.wrap_socket(s, server_hostname=self.host)
        return s

    def http(self, method, path, body=None, headers=None,
             response_timeout=READ_TIMEOUT):
        """A well-formed request through http.client."""
        if self.tls:
            conn = http.client.HTTPSConnection(
                self.host, self.port, timeout=CONNECT_TIMEOUT, context=self.ctx)
        else:
            conn = http.client.HTTPConnection(
                self.host, self.port, timeout=CONNECT_TIMEOUT)
        hdrs = {"Accept": "application/json, text/html",
                "Connection": "close"}
        if body is not None:
            hdrs["Content-Type"] = "application/json"
        hdrs.update(headers or {})
        try:
            conn.connect()
            conn.sock.settimeout(READ_TIMEOUT)
            conn.request(method, self.path(path), body=body, headers=hdrs)
            conn.sock.settimeout(response_timeout)
            t0 = time.monotonic()
            r = conn.getresponse()
            data = r.read()
            return Response(r.status, r.getheaders(), data,
                            time.monotonic() - t0)
        except socket.timeout:
            raise ProbeError("%s %s: no answer within %g s"
                             % (method, path, response_timeout))
        except (OSError, http.client.HTTPException) as e:
            raise ProbeError("%s %s: %s: %s"
                             % (method, path, e.__class__.__name__, e))
        finally:
            conn.close()

    def raw(self, method, path, headers, body=b"", timeout=READ_TIMEOUT):
        """A hand-built request over a plain socket, for malformed headers."""
        lines = ["%s %s HTTP/1.1" % (method, self.path(path)),
                 "Host: " + self.netloc, "Connection: close"] + list(headers)
        data = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + body
        try:
            s = self.socket(CONNECT_TIMEOUT)
        except OSError as e:
            raise ProbeError("connect: %s" % e)
        buf = b""
        t0 = time.monotonic()
        try:
            s.settimeout(timeout)
            s.sendall(data)
            while not _complete(buf):
                try:
                    chunk = s.recv(65536)
                except socket.timeout:
                    raise ProbeError("%s %s: no answer within %g s"
                                     % (method, path, timeout))
                except OSError as e:
                    if buf:          # reset after the answer: fine
                        break
                    raise ProbeError("%s %s: %s" % (method, path, e))
                if not chunk:
                    break
                buf += chunk
        finally:
            s.close()
        if not buf:
            raise ProbeError("%s %s: connection closed without an answer"
                             % (method, path))
        return _parse(buf, time.monotonic() - t0)


def _complete(buf):
    end = buf.find(b"\r\n\r\n")
    if end < 0:
        return False
    head = buf[:end].decode("latin-1").lower()
    m = re.search(r"\ncontent-length:\s*(\d+)", head)
    if m:
        return len(buf) - end - 4 >= int(m.group(1))
    if "transfer-encoding: chunked" in head:
        return buf.endswith(b"0\r\n\r\n")
    return False                     # wait for the server to close


def _parse(buf, elapsed):
    head, _, body = buf.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(None, 2)
    if len(parts) < 2 or not parts[1].isdigit():
        raise ProbeError("malformed status line %r" % lines[0][:80])
    headers = []
    for line in lines[1:]:
        key, _, value = line.partition(":")
        headers.append((key.strip(), value.strip()))
    if any(k.lower() == "transfer-encoding" and "chunked" in v.lower()
           for k, v in headers):
        body = _dechunk(body)
    return Response(int(parts[1]), headers, body, elapsed)


def _dechunk(body):
    out = b""
    while True:
        line, _, body = body.partition(b"\r\n")
        try:
            n = int(line.split(b";")[0].strip() or b"0", 16)
        except ValueError:
            break
        if n == 0:
            break
        out, body = out + body[:n], body[n + 2:]
    return out


# ------------------------------------------------------------- discovery --
def discover(t, tiny):
    """What a predict request needs: the unified server's dataset name, a
    class the server knows, and an upload id (a server started with
    --uploads-only holds no pool, so "index": 0 would fail first)."""
    info = {"dataset": None, "cls": "person", "upload": None}
    try:
        r = t.http("GET", "/api/datasets")
        if r.status == 200 and r.has("datasets") and r.json["datasets"]:
            info["dataset"] = r.json["datasets"][0]
    except ProbeError:
        pass
    query = "?dataset=" + info["dataset"] if info["dataset"] else ""
    try:
        r = t.http("GET", "/api/sample" + query)
        if r.has("classes") and r.json["classes"]:
            info["cls"] = r.json["classes"][0]
    except ProbeError:
        pass
    try:
        r = t.http("POST", "/api/upload", json.dumps(
            with_dataset({"image": b64(tiny), "name": "probe.png"}, info)))
        if r.status == 200 and r.has("upload") and r.json["upload"]:
            info["upload"] = r.json["upload"]
            if r.has("classes") and r.json["classes"]:
                info["cls"] = r.json["classes"][0]
    except ProbeError:
        pass
    return info


def with_dataset(req, info):
    if info["dataset"]:
        req["dataset"] = info["dataset"]
    return req


def predict_req(info, boxes):
    req = with_dataset({"boxes": boxes}, info)
    if info["upload"]:
        req["upload"] = info["upload"]
    else:
        req["index"] = 0
    return req


# ----------------------------------------------------------------- checks --
def check_body_cap(t):
    label = "1  POST /api/predict with Content-Length: -1"
    try:
        r = t.raw("POST", "/api/predict",
                  ["Content-Type: application/json", "Content-Length: -1"],
                  b"{}", timeout=BOMB_TIMEOUT * 2)
    except ProbeError as e:
        return report(False, label, str(e))
    ok = r.status in (400, 413) and not r.has("p_good")
    report(ok, label, "%s in %.2f s" % (r, r.elapsed))


def check_bomb(t, info):
    label = "2  POST /api/upload of a 16000x16000 PNG"
    t0 = time.monotonic()
    data = big_png()
    note("built the %dx%d PNG: %d bytes in %.1f s"
         % (BIG_SIDE, BIG_SIDE, len(data), time.monotonic() - t0))
    body = json.dumps(with_dataset({"image": b64(data), "name": "big.png"},
                                   info))
    try:
        r = t.http("POST", "/api/upload", body, response_timeout=BOMB_TIMEOUT)
    except ProbeError as e:
        return report(False, label, str(e))
    problems = []
    if not 400 <= r.status < 500:
        problems.append("status %d" % r.status)
    if not r.error:
        problems.append("no JSON error text")
    if leak(r.error):
        problems.append("error text leaks %r" % leak(r.error))
    if r.elapsed > BOMB_TIMEOUT:
        problems.append("answered after %.1f s" % r.elapsed)
    report(not problems, label,
           "%s in %.2f s" % (r, r.elapsed) if not problems
           else "; ".join(problems) + " (%s in %.2f s)" % (r, r.elapsed))


def check_empty_upload(t, info):
    label = "3  POST /api/upload {}"
    try:
        r = t.http("POST", "/api/upload", json.dumps(with_dataset({}, info)))
    except ProbeError as e:
        return report(False, label, str(e))
    ok = r.status == 400 and "no image data" in r.error.lower()
    report(ok, label, str(r))


def check_predict_inputs(t, info):
    cls = info["cls"]
    box = {"cls": cls, "box": [0, 0, 1, 1]}
    cases = [
        ("4a POST /api/predict with %d boxes" % (MAX_BOXES + 1),
         json.dumps(predict_req(info, [box] * (MAX_BOXES + 1))),
         lambda r: r.status == 400 and "too many boxes" in r.error.lower()),
        ("4b POST /api/predict with a NaN coordinate",
         json.dumps(predict_req(info, [{"cls": cls,
                                        "box": [float("nan"), 0, 1, 1]}])),
         lambda r: r.status == 400),
        ("4c POST /api/predict with \"boxes\": \"x\"",
         json.dumps(predict_req(info, "x")),
         lambda r: r.status == 400),
        ("4d POST /api/predict with a non-JSON body",
         "not json {",
         lambda r: r.status == 400),
    ]
    for label, body, good in cases:
        try:
            r = t.http("POST", "/api/predict", body)
        except ProbeError as e:
            report(False, label, str(e))
            continue
        problems = []
        if not good(r):
            problems.append("unexpected answer")
        if r.has("p_good"):
            problems.append("a verdict came back")
        if leak(r.error):
            problems.append("error text leaks %r" % leak(r.error))
        report(not problems, label,
               str(r) if not problems else "; ".join(problems) + " (%s)" % r)
    label = "4e POST /api/predict with Content-Length: abc"
    try:
        r = t.raw("POST", "/api/predict",
                  ["Content-Type: application/json", "Content-Length: abc"],
                  b"{}", timeout=BOMB_TIMEOUT * 2)
    except ProbeError as e:
        return report(False, label, str(e))
    ok = r.status == 400 and not r.has("p_good") and not leak(r.error)
    report(ok, label, str(r))


def check_routes(t):
    for label, method, path, good in [
            ("5a HEAD /", "HEAD", "/",
             lambda r: r.status == 200 and not r.body),
            ("5b GET /?view=api", "GET", "/?view=api",
             lambda r: r.status == 200
             and "text/html" in r.header("content-type").lower()),
            ("5c GET /nope", "GET", "/nope", lambda r: r.status == 404)]:
        try:
            r = t.http(method, path)
        except ProbeError as e:
            report(False, label, str(e))
            continue
        report(good(r), label, "%d %s" % (r.status, r.header("content-type")))


def check_headers(t, info):
    try:
        page = t.http("GET", "/")
        api = t.http("GET", "/api/sample"
                     + ("?dataset=" + info["dataset"] if info["dataset"]
                        else ""))
        err = t.http("POST", "/api/upload", json.dumps(with_dataset({}, info)))
    except ProbeError as e:
        for sub in "abcd":
            report(False, "6%s response headers" % sub, str(e))
        return
    both = [("/", page), ("/api/sample", api)]
    api_only = [("/api/sample", api), ("/api/upload", err)]

    def value(name, pairs):
        return ", ".join("%s: %s" % (p, r.header(name) or "(absent)")
                         for p, r in pairs)

    servers = [r.header("server") for _, r in both]
    ok = all("python" not in v.lower() and "basehttp" not in v.lower()
             for v in servers)
    report(ok, "6a Server header", value("server", both))
    ok = all(r.header("x-content-type-options").lower() == "nosniff"
             for _, r in both)
    report(ok, "6b X-Content-Type-Options", value("x-content-type-options", both))
    ok = all(r.header("x-frame-options").upper() == "DENY" for _, r in both)
    report(ok, "6c X-Frame-Options", value("x-frame-options", both))
    ok = all("no-store" in r.header("cache-control").lower()
             for _, r in api_only)
    report(ok, "6d Cache-Control on /api/*", value("cache-control", api_only))


def check_idle(t, count, wait):
    label = "7  idle connections closed by the server"
    partial = ("GET %s HTTP/1.1\r\nHost: %s\r\n"
               % (t.path("/"), t.netloc)).encode("ascii")   # never finished
    opened, failed = [], []
    for i in range(count):
        try:
            s = t.socket(CONNECT_TIMEOUT)
            if i % 2:
                s.sendall(partial)
            opened.append(s)
        except OSError as e:
            failed.append(str(e))
        time.sleep(0.02)        # a burst overflows the listen backlog
    if not opened:
        return report(False, label, "could not open any socket: %s"
                      % failed[-1])
    note("%d sockets open%s, waiting %g s"
         % (len(opened), " (%d connects failed)" % len(failed) if failed
            else "", wait))
    time.sleep(wait)
    # a closed socket is readable (EOF, or an answer followed by EOF, or
    # a reset); one that stays silent for 3 s is still open
    still, got, refused = list(opened), {s: b"" for s in opened}, 0
    for s in opened:
        s.setblocking(False)
    deadline = time.monotonic() + 3.0
    while still and time.monotonic() < deadline:
        readable, _, _ = select.select(still, [], [],
                                       max(0.0, deadline - time.monotonic()))
        for s in readable:
            try:
                chunk = s.recv(4096)
                while chunk and getattr(s, "pending", lambda: 0)():
                    chunk += s.recv(4096)
            except (ssl.SSLWantReadError, ssl.SSLWantWriteError):
                continue
            except OSError:
                chunk = b""          # reset by the server: closed
            if chunk:
                got[s] += chunk
                continue
            still.remove(s)
            if got[s].startswith(b"HTTP/") and b" 503 " in got[s][:16]:
                refused += 1
    for s in opened:
        s.close()
    closed = len(opened) - len(still)
    detail = "%d/%d closed" % (closed, len(opened))
    if refused:
        detail += ", %d of them refused with 503 at connect" % refused
    if still:
        detail += ", %d still open after %g s" % (len(still), wait)
    if failed:
        detail += ", %d connects failed (%s)" % (len(failed), failed[-1])
    report(not still, label, detail)


def check_happy(t, info, tiny):
    label = "8a POST /api/upload of a 4x4 PNG"
    try:
        r = t.http("POST", "/api/upload", json.dumps(
            with_dataset({"image": b64(tiny), "name": "probe.png"}, info)))
    except ProbeError as e:
        report(False, label, str(e))
        return report(False, "8b POST /api/predict on it", "no upload")
    uid = r.json["upload"] if r.has("upload") else None
    ok = r.status == 200 and isinstance(uid, str) and bool(uid)
    report(ok, label, "%d, upload id %s" % (r.status, "present" if ok
                                            else "missing") if ok else str(r))
    label = "8b POST /api/predict on it"
    if not ok:
        return report(False, label, "no upload id to score")
    cls = info["cls"]
    if r.has("classes") and r.json["classes"]:
        cls = r.json["classes"][0]
    req = with_dataset({"upload": uid,
                        "boxes": [{"cls": cls, "box": [1, 1, 3, 3]}]}, info)
    try:
        r = t.http("POST", "/api/predict", json.dumps(req))
    except ProbeError as e:
        return report(False, label, str(e))
    ok = (r.status == 200 and r.has("p_good")
          and isinstance(r.json["p_good"], (int, float)))
    report(ok, label, "200 p_good=%.3f (%s)" % (r.json["p_good"], cls)
           if ok else str(r))


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(
        description="Probe an interactive demo server before opening it "
                    "to the public; see the module docstring for the checks.")
    ap.add_argument("url", help="e.g. http://127.0.0.1:7864 or "
                                "https://demo.example.org/paper/")
    ap.add_argument("--idle", action="store_true",
                    help="also run the idle-connection check (takes "
                         "--timeout seconds)")
    ap.add_argument("--connections", type=int, default=40,
                    help="sockets to open for --idle (default 40, above "
                         "the servers' cap of 32)")
    ap.add_argument("--timeout", type=float, default=45,
                    help="seconds to wait in --idle before requiring the "
                         "sockets closed (default 45; above the servers' "
                         "30 s, raise it above a proxy's own timeouts)")
    ap.add_argument("--skip-happy", action="store_true",
                    help="skip check 8 (upload + predict), for a server "
                         "with no model behind it")
    ap.add_argument("--insecure", action="store_true",
                    help="do not verify the TLS certificate")
    args = ap.parse_args()

    t = Target(args.url, args.insecure)
    print("probe " + args.url, flush=True)
    try:
        t.socket(CONNECT_TIMEOUT).close()
    except OSError as e:
        report(False, "0  connect to %s:%d" % (t.host, t.port), str(e))
        return 1
    tiny = tiny_png()
    info = discover(t, tiny)
    note("predict requests use %s, class %r%s"
         % ("dataset %r" % info["dataset"] if info["dataset"]
            else "no dataset field", info["cls"],
            ", a held upload" if info["upload"] else ", index 0"))

    check_body_cap(t)
    check_bomb(t, info)
    check_empty_upload(t, info)
    check_predict_inputs(t, info)
    check_routes(t)
    check_headers(t, info)
    if args.idle:
        check_idle(t, args.connections, args.timeout)
    else:
        print("skip  7  idle connections (pass --idle)", flush=True)
    if args.skip_happy:
        print("skip  8  happy path (--skip-happy)", flush=True)
    else:
        check_happy(t, info, tiny)

    failed = RESULTS.count(False)
    print("%d checks, %d passed, %d failed"
          % (len(RESULTS), len(RESULTS) - failed, failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
