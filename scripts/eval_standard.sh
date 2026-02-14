#!/bin/bash
# ============================================================================
# Standard Evaluation (GUI-AIMA baseline)
# ============================================================================
# Uses GUI-AIMA's original inference() pipeline.
# Supports single-round and multi-round (crop_resize / token_concat).
#
# Usage:
#   bash scripts/eval_standard.sh                           # default: single-round high-res
#   bash scripts/eval_standard.sh /path/to/checkpoint       # evaluate a checkpoint
#   ROUNDS=2 STRATEGY=crop_resize bash scripts/eval_standard.sh
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
ROUNDS="${ROUNDS:-1}"
STRATEGY="${STRATEGY:-none}"       # none / crop_resize / token_concat
RESOLUTION="${RESOLUTION:-high}"   # high / low
CROP_RATIO="${CROP_RATIO:-0.3}"
MAX_SAMPLES="${MAX_SAMPLES:-}"     # empty = all samples

# ── Run ─────────────────────────────────────────────────────────────────────
export PYTHONPATH="${GUI_AIMA_SRC}:${PYTHONPATH}"
cd "${PROJECT_DIR}"

MODEL_NAME=$(basename "${MODEL}")
if [ "${ROUNDS}" -gt 1 ]; then
    TAG="r${ROUNDS}_${STRATEGY}_${RESOLUTION}"
else
    TAG="single_${RESOLUTION}"
fi
SAVE_PATH="${SAVE_BASE}/${MODEL_NAME}/${TAG}"

echo "=== Standard Evaluation ==="
echo "  model:      ${MODEL}"
echo "  rounds:     ${ROUNDS}"
echo "  strategy:   ${STRATEGY}"
echo "  resolution: ${RESOLUTION}"
echo "  save_path:  ${SAVE_PATH}"
echo ""

CMD="python eval/eval_screenspot_pro.py \
    --model_name_or_path ${MODEL} \
    --data_path ${DATA_PATH} \
    --save_path ${SAVE_PATH} \
    --rounds ${ROUNDS} \
    --multi_strategy ${STRATEGY} \
    --resolution ${RESOLUTION} \
    --crop_ratio ${CROP_RATIO}"

if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

eval "${CMD}"
