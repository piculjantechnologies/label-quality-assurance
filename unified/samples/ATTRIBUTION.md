# Demo sample images — provenance and attribution

The five demo photographs are from the COCO 2017 *train* split (the paper's
test split — disjoint from this release's training data). They are **not
redistributed** in this folder; `python3 samples/fetch_samples.py` downloads
them from [images.cocodataset.org](https://cocodataset.org). Each image is a
Flickr photograph under its own Creative Commons license:

| sample | COCO train2017 id | Flickr photo id | Flickr source (COCO's `flickr_url`) | license |
| --- | --- | --- | --- | --- |
| 1 | 000000566046 | 3785960268 | [flickr](http://farm3.staticflickr.com/2648/3785960268_28d85f5518_z.jpg) | [CC BY-NC-SA 2.0](http://creativecommons.org/licenses/by-nc-sa/2.0/) |
| 2 | 000000293377 | 5247651709 | [flickr](http://farm6.staticflickr.com/5170/5247651709_38d137de1b_z.jpg) | [CC BY-NC-ND 2.0](http://creativecommons.org/licenses/by-nc-nd/2.0/) |
| 3 | 000000508985 | 8530368974 | [flickr](http://farm9.staticflickr.com/8530/8530368974_d90937655b_z.jpg) | [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/) |
| 4 | 000000161386 | 101357137 | [flickr](http://farm1.staticflickr.com/42/101357137_101cbf3976_z.jpg) | [CC BY-NC-SA 2.0](http://creativecommons.org/licenses/by-nc-sa/2.0/) |
| 5 | 000000436694 | 8252056817 | [flickr](http://farm9.staticflickr.com/8062/8252056817_ba7c9154ec_z.jpg) | [CC BY 2.0](http://creativecommons.org/licenses/by/2.0/) |

The Flickr URLs and license ids are COCO's own records for these images
and are kept as such; checked on 27 August 2026, the links for samples 2,
3 and 4 no longer resolve. COCO does not record photographer names, so
the author credit each license asks for can only be recovered from the
Flickr photo page where it still exists (photo ids above).

The `sample_N.json` label files are derived from the COCO annotations
(released under CC BY 4.0 by the COCO Consortium) and are included:

- **samples 1, 3, 4** are the COCO ground-truth annotations *verbatim*
  (1, 29, and 12 boxes respectively — including sample 4's two poster
  birds, which are COCO's own `bird` labels);
- **sample 2** is the ground truth with a hand-made corruption: the pizza
  box's top edge is raised 100 px and its right edge extended 20 px
  (154 × 42 → 174 × 142 px) and its class swapped to `dog`;
- **sample 5** is the ground truth with hand-made corruptions of all three
  boxes (car bottom raised to y2 351.2, both traffic lights widened, one
  shifted).

The Good/Bad targets used by `qa_demo.py` are assigned by this release:
samples 1/3/4 = Good because they are the untouched ground truth,
samples 2/5 = Bad because of the corruptions above.
