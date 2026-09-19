#!/bin/bash
# End-to-end unified-pipeline training run -- this script is the
# recipe's only carrier (bare qa_train.py defaults are NOT the release
# recipe). Dataset comes from QA_DATASET (coco | voc) plus its data
# environment variables; see dataset.py.
#
#   QA_DATASET=voc  VOC_POOL=...                         ./run_qa.sh
#   QA_DATASET=coco COCO_IMAGES=... COCO_ANNOTATIONS=... ./run_qa.sh
#
# The recipe's effective batch is 64 with micro-batch-8 BatchNorm
# statistics. The default runs it as one physical batch of 64 with
# GhostBatchNorm supplying those statistics (~30 GB of GPU memory);
# FAITHFUL=1 runs the same effective-batch recipe as 8 x 8 accumulation for
# small GPUs. Resumes from $OUT/checkpoint.pth after an interruption.
cd "$(dirname "$0")" || exit 1
: "${QA_DATASET:?set QA_DATASET=coco or QA_DATASET=voc}"
OUT="${OUT:-artifacts_$QA_DATASET}"
mkdir -p "$OUT"
if [ -n "$FAITHFUL" ]; then
    ARGS="--batch_size 8 --accum 8"
else
    ARGS="--batch_size 64 --accum 1 --ghost_bn 8 --workers ${WORKERS:-20} \
        --cache_images"
fi
ATTEMPTS="${RETRIES:-20}"
for attempt in $(seq 1 "$ATTEMPTS"); do
    python3 qa_train.py \
        --epochs 60 $ARGS --amp \
        --lr 1e-3 --weight_decay 1e-2 --warmup_epochs 2 \
        --finetune_mult 0.1 --freeze_through layer2 \
        --ema_decay 0.998 --aux_weight 0.3 \
        --p_exact 0.5 --detector_share 0.1 --ih_share 0.5 --compose 0.5 \
        --severity_mix balanced --aug_prob 0.5 --seed 0 \
        --p_edge 0.25 --swap_share 0.1 \
        --per_box_weight 0.3 --margin_weight 0.3 \
        --image_trunk resnet50 \
        --out "$OUT" --resume 2>&1 | tee "$OUT/attempt.out" \
        | tee -a "$OUT/train.out"
    status=${PIPESTATUS[0]}
    [ "$status" -eq 0 ] && exit 0
    echo "[run] train exited $status (attempt $attempt/$ATTEMPTS)" \
        | tee -a "$OUT/train.out"
    # a configuration error repeats identically; only retry crashes
    grep -qE "error: (unrecognized|argument)|No such file|FileNotFoundError|set COCO_|set VOC_" \
        "$OUT/attempt.out" \
        && { echo "[run] configuration error, not retrying" \
             | tee -a "$OUT/train.out"; exit 1; }
    sleep 60
done
echo "[run] giving up after $ATTEMPTS attempts" | tee -a "$OUT/train.out"
exit 1
