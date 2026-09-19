#!/bin/bash
# End-to-end training run — this script is the recipe's only carrier
# (bare train_coco_corr.py defaults are NOT the release recipe; the
# module docstring of train_coco_corr.py maps every flag below to its
# README item).
#
#   export COCO_IMAGES=/path/to/coco/val2017
#   export COCO_ANNOTATIONS=/path/to/coco/annotations/instances_val2017.json
#   ./run.sh            # the released configuration (~23 GB GPU, observed)
#   FAITHFUL=1 ./run.sh # the same recipe, as 8 x 8 accumulation
#
# Both train the paper's effective batch of 64, with BatchNorm statistics
# per 8 samples (README item 8), and sparse plane transport
# (bit-identical batches). The default runs it as 32 x 2 accumulation
# with GhostBatchNorm supplying the per-8 statistics; FAITHFUL runs it as 8 x 8
# accumulation for small GPUs. 160 epochs, no early stopping; the fixed
# validation set is 10% of val2017 (454 images, draw rounds 0-3, 3632
# candidates) drawn as evaluate.py draws its test pool, and
# $OUT/best_model.pth is the epoch with the highest validation ROC-AUC
# (EMA weights, README item 7). Resumes from $OUT/checkpoint.pth after an
# interruption. refit.py then fits the verdict on the same held-out
# val2017 images, draw rounds 4-11 (README item 10), and writes
# $OUT/best_model_refit.pth, the released checkpoint.
#
# Environment: OUT (output directory, default artifacts), WORKERS (loader
# workers, default 22, or 8 with FAITHFUL), RETRIES (training attempts
# before giving up, default 20; each after a crash resumes from
# $OUT/checkpoint.pth after 60 s), FAITHFUL (any non-empty value).
# With the default OUT, training appends to artifacts/train.out and the
# run replaces the shipped records and best_model_refit.pth.
cd "$(dirname "$0")" || exit 1
# one thread per process: the loader workers otherwise oversubscribe the CPUs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
: "${COCO_IMAGES:?set COCO_IMAGES to the COCO image directory}"
: "${COCO_ANNOTATIONS:?set COCO_ANNOTATIONS to the COCO instances json}"
OUT="${OUT:-artifacts}"
mkdir -p "$OUT"
if [ -n "$FAITHFUL" ]; then
    SPEED_ARGS="--batch_size 8 --accum 8 --sparse_planes \
        --workers ${WORKERS:-8}"
else
    SPEED_ARGS="--batch_size 32 --accum 2 --ghost_bn 8 --sparse_planes \
        --workers ${WORKERS:-22}"
fi
trained=0
for attempt in $(seq 1 "${RETRIES:-20}"); do
    if python3 train_coco_corr.py \
        --images "$COCO_IMAGES" --annotations "$COCO_ANNOTATIONS" \
        --epochs 160 $SPEED_ARGS --amp --patience 0 \
        --optimizer adamw --lr 1e-3 --weight_decay 1e-2 --img_lr_mult 0.1 \
        --warmup_epochs 2 --cosine --ema_decay 0.998 \
        --input_norm --freeze_image_bn \
        --augment_p 0.1 --detector_share 0.1 --swap_share 0.1 \
        --hard_negatives --paper_mix 0.5 --p_exact 0.5 \
        --val_split 0.1 --val_rounds 4 --val_test_protocol \
        --pretrained_image --corr_masked --corr_max --corr_cos \
        --raster_head --box_weight 0.5 --cls_weight 0.5 \
        --seed 0 --out "$OUT" --resume >> "$OUT/train.out" 2>&1; then
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
python3 refit.py "$OUT/best_model.pth" \
    --images "$COCO_IMAGES" --annotations "$COCO_ANNOTATIONS" \
    --val_split 0.1 --seed 0 \
    --out "$OUT/best_model_refit.pth" --report "$OUT/refit.json" \
    > "$OUT/refit.out" 2>&1
