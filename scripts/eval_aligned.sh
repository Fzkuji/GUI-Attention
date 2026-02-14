#!/bin/bash
# ============================================================================
# Aligned Evaluation (matches training pipeline)
# ============================================================================
# Uses the same attention extraction + prediction method as training.
# Multi-round with token concatenation, attention argmax/region prediction.
#
# Usage:
#   bash scripts/eval_aligned.sh                           # default: 5 rounds, region
#   bash scripts/eval_aligned.sh /path/to/checkpoint       # evaluate a checkpoint
#   ROUNDS=3 PRED=argmax bash scripts/eval_aligned.sh      # override config
# ============================================================================

set -e

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR="/mnt/data/zichuanfu"
PROJECT_DIR="${BASE_DIR}/GUI-Attention"
GUI_AIMA_SRC="${BASE_DIR}/Experiments/GUI-AIMA/src"

MODEL="${1:-${BASE_DIR}/models/GUI-AIMA-3B}"
DATA_PATH="${BASE_DIR}/data/ScreenSpot-Pro"
SAVE_BASE="${BASE_DIR}/results/screenspot_pro"

# ── Config (override via environment variables) ─────────────────────────────
ROUNDS="${ROUNDS:-5}"
CROP_RATIO="${CROP_RATIO:-0.3}"
PRED="${PRED:-region}"             # region (recommended) / argmax
MAX_SAMPLES="${MAX_SAMPLES:-}"     # empty = all samples

# ── Run ─────────────────────────────────────────────────────────────────────
export PYTHONPATH="${GUI_AIMA_SRC}:${PYTHONPATH}"
cd "${PROJECT_DIR}"

MODEL_NAME=$(basename "${MODEL}")
TAG="aligned_r${ROUNDS}_crop${CROP_RATIO}_${PRED}"
SAVE_PATH="${SAVE_BASE}/${MODEL_NAME}/${TAG}"

echo "=== Aligned Evaluation ==="
echo "  model:      ${MODEL}"
echo "  rounds:     ${ROUNDS}"
echo "  crop_ratio: ${CROP_RATIO}"
echo "  prediction: ${PRED}"
echo "  save_path:  ${SAVE_PATH}"
echo ""

CMD="python eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path ${MODEL} \
    --data_path ${DATA_PATH} \
    --save_path ${SAVE_PATH} \
    --mode aligned \
    --rounds ${ROUNDS} \
    --crop_ratio ${CROP_RATIO} \
    --prediction_method ${PRED}"

if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

eval "${CMD}"
