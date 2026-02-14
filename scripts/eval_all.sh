#!/bin/bash
# ============================================================================
# Full Evaluation Pipeline
# ============================================================================
# Runs all evaluation configurations for a given model.
# Covers: standard baselines + aligned multi-round + ablations.
#
# Usage:
#   bash scripts/eval_all.sh /path/to/model
#   MAX_SAMPLES=50 bash scripts/eval_all.sh /path/to/model   # quick test
# ============================================================================

set -e

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR="/mnt/data/zichuanfu"
PROJECT_DIR="${BASE_DIR}/GUI-Attention"
GUI_AIMA_SRC="${BASE_DIR}/Experiments/GUI-AIMA/src"

MODEL="${1:-${BASE_DIR}/models/GUI-AIMA-3B}"
DATA_PATH="${BASE_DIR}/data/ScreenSpot-Pro"
SAVE_BASE="${BASE_DIR}/results/screenspot_pro"
MAX_SAMPLES="${MAX_SAMPLES:-}"

export PYTHONPATH="${GUI_AIMA_SRC}:${PYTHONPATH}"
cd "${PROJECT_DIR}"

MODEL_NAME=$(basename "${MODEL}")
SAMPLE_FLAG=""
if [ -n "${MAX_SAMPLES}" ]; then
    SAMPLE_FLAG="--max_samples ${MAX_SAMPLES}"
fi

echo "================================================================"
echo "  Full Evaluation: ${MODEL_NAME}"
echo "  Data: ${DATA_PATH}"
if [ -n "${MAX_SAMPLES}" ]; then
    echo "  Max samples: ${MAX_SAMPLES} (quick test mode)"
fi
echo "================================================================"
echo ""

# ── 1. Standard baselines (GUI-AIMA inference pipeline) ─────────────────────

echo ">>> [1/6] Standard single-round, HIGH res"
python eval/eval_screenspot_pro.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/single_high" \
    --rounds 1 --resolution high \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

echo ">>> [2/6] Standard single-round, LOW res"
python eval/eval_screenspot_pro.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/single_low" \
    --rounds 1 --resolution low \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

echo ">>> [3/6] Standard 2-round crop_resize, HIGH res"
python eval/eval_screenspot_pro.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/r2_crop_resize_high" \
    --rounds 2 --multi_strategy crop_resize --resolution high \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

# ── 2. Aligned multi-round (our method) ────────────────────────────────────

echo ">>> [4/6] Aligned 5-round, region prediction (our method)"
python eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/aligned_r5_crop0.3_region" \
    --mode aligned --rounds 5 --crop_ratio 0.3 --prediction_method region \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

echo ">>> [5/6] Aligned 3-round, region prediction"
python eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/aligned_r3_crop0.3_region" \
    --mode aligned --rounds 3 --crop_ratio 0.3 --prediction_method region \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

echo ">>> [6/6] Aligned 5-round, argmax prediction (ablation)"
python eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/aligned_r5_crop0.3_argmax" \
    --mode aligned --rounds 5 --crop_ratio 0.3 --prediction_method argmax \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

echo "================================================================"
echo "  All evaluations complete. Results in:"
echo "  ${SAVE_BASE}/${MODEL_NAME}/"
echo "================================================================"
