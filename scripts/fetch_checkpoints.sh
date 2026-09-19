#!/bin/bash
# Download the model checkpoints from Azure Blob Storage. None of the four
# (~523 MB together) is in the git repository; a file that already matches
# its recorded sha256 is skipped.
#
# By default the files come from the public container of this release,
# https://labelqa.blob.core.windows.net/checkpoints/v1.0. Set
# LABELQA_CHECKPOINT_URL to fetch them from a mirror instead (the base URL of
# the container or virtual directory that holds them; a read-only SAS token
# may follow it as a query string). Each file is verified against the sha256
# recorded in scripts/checkpoints.json. Blobs are named after each path with
# "/" replaced by "__".
#
#   bash scripts/fetch_checkpoints.sh
set -eu

BASE="${LABELQA_CHECKPOINT_URL:-https://labelqa.blob.core.windows.net/checkpoints/v1.0}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

python3 - "$BASE" <<'PY'
import hashlib, json, os, sys, urllib.request

base = sys.argv[1]
# A base URL may carry a query string (e.g. an Azure SAS token), in which case
# the object name goes before the "?", not after it.
prefix, _, query = base.partition("?")
prefix = prefix.rstrip("/")

for entry in json.load(open("scripts/checkpoints.json")):
    path, want = entry["path"], entry["sha256"]
    if os.path.exists(path) and hashlib.sha256(open(path, "rb").read()).hexdigest() == want:
        print(f"ok (cached)  {path}")
        continue
    os.makedirs(os.path.dirname(path), exist_ok=True)
    name = path.replace("/", "__")
    url = f"{prefix}/{name}?{query}" if query else f"{prefix}/{name}"
    print(f"fetching     {path}  <- {url.split(chr(63))[0]}")
    urllib.request.urlretrieve(url, path)
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != want:
        raise SystemExit(f"checksum mismatch for {path}\n  expected {want}\n  got      {got}")
    print(f"ok           {path}")
print("all checkpoints present and verified")
PY
