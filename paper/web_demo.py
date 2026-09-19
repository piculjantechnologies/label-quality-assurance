"""Gradio web demo: score any image + label pair with the released model.

Mirrors the paper's published demo UI: upload an image and a label as JSON
(class name -> list of [x1, y1, x2, y2] boxes), get the drawn label and
P(Good)/P(Bad). The five README samples appear as click-to-run examples
once their images are fetched (`python3 samples/fetch_samples.py`).
`--no-upload` turns the page into an examples-only demo: the image input
accepts no upload, drop, webcam or paste, and a click on one of the five
examples is the only way to load a photo (the label JSON stays editable;
the HTTP API is unchanged).

    pip install "gradio>=6"
    python3 web_demo.py [--checkpoint artifacts/best_model_refit.pth] [--share] [--no-upload]

--share tunnels this machine's demo through gradio's public relay (the
link is valid for up to a week) and is meant for short sessions; a
permanent public instance binds to localhost behind a reverse proxy that
terminates TLS and rate-limits (see the README). There is no login.
Uploads are capped at 20 MB and 25 megapixels, and gradio's upload cache
is cleared hourly.
"""

import argparse
import colorsys
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
import gradio as gr

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from coco_corrnet import CorrNet
from data_loader import get_sample
from interactive_demo import COCO_CLASSES

DESCRIPTION = """
<p>Given an image and a candidate bounding-box label, the model decides
whether the label is <b>good</b> (every ground-truth object matched by one
box of the right class inside its uncertainty region) or <b>bad</b>
(anything else). Method: Pičuljan &amp; Car,
<a href="https://doi.org/10.3390/app13106234">Applied Sciences
13(10):6234, 2023</a>, with the implementation details documented in the README of the
<a href="{readme}">code release</a> (folder <code>paper/</code>).
Labels are JSON: <code>{{"class name": [[x1, y1, x2, y2], ...]}}</code>
with the 80 COCO class names. The examples below are that README's five
demo samples (targets Good, Bad, Good, Good, Bad).</p>
"""
README_URL = ("https://github.com/piculjantechnologies/label-quality-assurance"
              "/blob/main/paper/README.md")
NO_UPLOAD_NOTE = """
<p><b>Click one of the examples below to load its photo</b> — this public
instance takes no uploaded images. The label JSON can be edited before
pressing Submit.</p>
"""


def _distinct_colors(n):
    return [tuple(int(x * 255) for x in colorsys.hsv_to_rgb(i / (n + 1), 1, 1))
            for i in range(n)]


COLORS = _distinct_colors(80)
NAME2IDX = {n: i for i, n in enumerate(COCO_CLASSES)}
MAX_PIXELS = 25_000_000
COORD_LIMIT = 1e6
LABEL_SHAPE = 'a JSON object {"class name": [[x1, y1, x2, y2], ...]}'
CREDIT = ("Sample photographs: COCO 2017 (Flickr photographs under their "
          "individual Creative Commons licences; annotations CC BY 4.0) — "
          "see samples/ATTRIBUTION.md.")


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def parse_label(label_json):
    """The label JSON -> {class_index: [[x, y, w, h], ...]}, or gr.Error.

    The documented shape only: an object mapping COCO class names to lists
    of [x1, y1, x2, y2] boxes of four finite numbers (clipped to
    +/-COORD_LIMIT). The message never carries raw exception text.
    """
    if isinstance(label_json, str):
        try:
            payload = json.loads(label_json or "{}")
        except ValueError:
            raise gr.Error(f"Label is not valid JSON: expected {LABEL_SHAPE}")
    else:
        payload = label_json or {}
    if not isinstance(payload, dict):
        raise gr.Error(f"Label must be {LABEL_SHAPE}")
    unknown = sorted(str(k)[:40] for k in payload if k not in NAME2IDX)
    if unknown:
        raise gr.Error(f"Unknown class name(s): {', '.join(unknown)}")
    annotation = {}
    for name, boxes in payload.items():
        if not isinstance(boxes, list):
            raise gr.Error(f"{name}: expected a list of [x1, y1, x2, y2] boxes")
        for box in boxes:
            if (not isinstance(box, list) or len(box) != 4
                    or not all(_is_number(c) and math.isfinite(c)
                               for c in box)):
                raise gr.Error(f"{name}: every box must be four finite "
                               f"numbers [x1, y1, x2, y2]")
            x1, y1, x2, y2 = [min(COORD_LIMIT, max(-COORD_LIMIT, float(c)))
                              for c in box]
            annotation.setdefault(NAME2IDX[name], []).append(
                [x1, y1, x2 - x1, y2 - y1])
    return annotation


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",
                    default=os.path.join(_HERE, "artifacts",
                                         "best_model_refit.pth"))
    ap.add_argument("--readme-url", default=README_URL,
                    help="where the page's 'code release' link points "
                         "(the paper/README.md of the public repository)")
    ap.add_argument("--no-upload", action="store_true",
                    help="examples-only page: the image input takes no "
                         "upload, drop, webcam or paste; photos come from "
                         "the five examples (the label stays editable)")
    ap.add_argument("--share", action="store_true",
                    help="tunnel this machine's demo through gradio's public "
                         "relay (link valid up to a week; for short "
                         "sessions — a permanent public instance binds to "
                         "localhost behind a reverse proxy)")
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        sys.exit(f"checkpoint not found: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.checkpoint} on {device}")
    model = CorrNet.from_checkpoint(args.checkpoint, device)
    model.eval()

    def classify(image, label_json):
        if image is None:
            raise gr.Error("Upload an image first.")
        h, w = image.shape[:2]
        if h * w > MAX_PIXELS:
            raise gr.Error("image too large")
        annotation = parse_label(label_json)

        image = np.ascontiguousarray(image)
        bgr = image[:, :, ::-1].astype(np.uint8)

        cats, background = get_sample(640, bgr, annotation)
        cats = torch.from_numpy(cats).float().unsqueeze(0).to(device)
        background = torch.from_numpy(background).float().unsqueeze(0).to(device)
        with torch.no_grad():
            p = torch.softmax(model(cats, background), 1)

        for idx, boxes in annotation.items():
            for x, y, w, h in boxes:
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(image, (x, y), (x + w, y + h), COLORS[idx])
                cv2.putText(image, COCO_CLASSES[idx], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx])
        return image, {"Good Label": p[0][1].item(),
                       "Bad Label": p[0][0].item()}

    samples_dir = os.path.join(_HERE, "samples")
    examples = []
    for n in range(1, 6):
        png = os.path.join(samples_dir, f"sample_{n}.png")
        js = os.path.join(samples_dir, f"sample_{n}.json")
        if os.path.exists(png) and os.path.exists(js):
            with open(js) as f:
                examples.append([png, f.read()])
    if not examples:
        print("no example images found — run `python3 samples/fetch_samples.py` "
              "to offer the five README samples as click-to-run examples")

    default_label = examples[0][1] if examples else json.dumps(
        {"person": [[100, 100, 200, 300]]}, indent=1)
    attribution = os.path.join(samples_dir, "ATTRIBUTION.md")
    if os.path.exists(attribution):
        with open(attribution) as f:
            credits_url = args.readme_url.replace(
                "README.md", "samples/ATTRIBUTION.md")
            article = (f"Credits for the five example photos, from the code "
                       f"release's [samples/ATTRIBUTION.md]({credits_url}):"
                       f"\n\n" + f.read())
    else:
        article = CREDIT

    description = DESCRIPTION.format(readme=args.readme_url)
    if args.no_upload:
        description += NO_UPLOAD_NOTE
    gr.Interface(
        title="Label quality assurance for object detection",
        description=description,
        article=article,
        fn=classify,
        inputs=[gr.Image(label="Raw Image", interactive=not args.no_upload),
                gr.Code(value=default_label, language="json", label="Label")],
        outputs=[gr.Image(label="Label Visualization"),
                 gr.Label(label="Label Quality Assurance")],
        examples=examples or None,
        delete_cache=(3600, 3600),       # gradio's upload cache, hourly
        flagging_mode="never",           # no Flag button, nothing stored
    ).launch(share=args.share, theme=gr.themes.Monochrome(),
             max_file_size="20mb")


if __name__ == "__main__":
    main()
