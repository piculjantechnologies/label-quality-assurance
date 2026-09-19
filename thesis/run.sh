#!/bin/bash
# End-to-end training run — the recipe of the released checkpoint.
#
#   ./run.sh            # trains on the small set (the VOC 2012 validation
#                       # split under ~/fiftyone/voc-2012/, or VOC_POOL),
#                       # then the held-out refit
#
# train.py's defaults are the recipe (effective batch 64 as 32 x 2
# accumulation with GhostBatchNorm over groups of 8, AdamW with the image
# branch at 0.1 x the rate (README item 5), warmup + cosine over 160 epochs,
# weight EMA and selection by validation ROC-AUC (README item 9), mixed
# precision; about 4 GB of GPU memory) except the per-cell head; this
# script adds the per-cell head and its two losses (README item 4), the
# small-set --pool, 20 loader workers (which fix which training candidates
# are drawn), seed 0 (the default, restated) and resuming: an interruption
# continues from $OUT/checkpoint.pth. Training writes $OUT/best_model.pth;
# refit.py then fits the verdict on the run's held-out validation images
# (README item 6) and writes $OUT/best_model_refit.pth, the released
# checkpoint.
#
# Environment: VOC_POOL (small-set split directory holding
# processed_annotations/ and data/, default ~/fiftyone/voc-2012/validation),
# OUT (output directory, default artifacts; train.out is appended to),
# WORKERS (loader workers, default 20), RETRIES (attempts before giving up,
# default 20; each retry resumes after 60 s).
cd "$(dirname "$0")" || exit 1
# one thread per process: the loader workers otherwise oversubscribe the CPUs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
POOL="${VOC_POOL:-$HOME/fiftyone/voc-2012/validation}/processed_annotations"
OUT="${OUT:-artifacts}"
mkdir -p "$OUT"
trained=0
for attempt in $(seq 1 "${RETRIES:-20}"); do
    if python3 train.py --pool "$POOL" \
        --raster_head --box_weight 0.5 --cls_weight 0.5 \
        --workers "${WORKERS:-20}" --seed 0 \
        --out "$OUT" --resume >> "$OUT/train.out" 2>&1; then
        trained=1
        break
    else
        rc=$?
    fi
    echo "[run] train exited $rc (attempt $attempt/${RETRIES:-20}); retrying in 60s" \
        >> "$OUT/train.out"
    sleep 60
done
if [ "$trained" != 1 ]; then
    echo "[run] giving up after ${RETRIES:-20} attempts" >> "$OUT/train.out"
    exit 1
fi
python3 refit.py "$OUT/best_model.pth" --pool "$POOL" --seed 0 \
    --out "$OUT/best_model_refit.pth" --report "$OUT/refit.json" \
    > "$OUT/refit.out" 2>&1
