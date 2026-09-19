"""Download the five demo images from COCO's image server (one-time, ~1 MB).

The demo samples' photographs are COCO train2017 images under individual
Flickr licenses (see ATTRIBUTION.md) and are not redistributed with this
folder; this script fetches the originals from images.cocodataset.org and
stores them as samples/sample_N.png, pixel-identical to the images the
released results were produced with.

    python3 samples/fetch_samples.py
"""

import os
import sys
import urllib.request

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

# sample number -> COCO train2017 image id (see ATTRIBUTION.md)
IMAGE_IDS = {1: 566046, 2: 293377, 3: 508985, 4: 161386, 5: 436694}
URL = "http://images.cocodataset.org/train2017/{:012d}.jpg"


def main():
    failures = 0
    for n, image_id in IMAGE_IDS.items():
        out = os.path.join(_HERE, f"sample_{n}.png")
        if os.path.isfile(out):
            print(f"sample_{n}.png already present, skipping")
            continue
        url = URL.format(image_id)
        print(f"fetching {url} -> samples/sample_{n}.png")
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failures += 1
            continue
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  FAILED: could not decode {url}", file=sys.stderr)
            failures += 1
            continue
        cv2.imwrite(out, image)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
