#!/bin/bash
# ============================================================================
# Full Evaluation Pipeline
# ============================================================================
# Runs all evaluation configurations for a given model.
# Covers: standard baselines + multi-precision foveation + ablations.
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

echo ">>> [1/7] Standard single-round, HIGH res"
python eval/eval_screenspot_pro.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/single_high" \
    --rounds 1 --resolution high \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

echo ">>> [2/7] Standard single-round, LOW res"
python eval/eval_screenspot_pro.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/single_low" \
    --rounds 1 --resolution low \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

# ── 2. Multi-precision foveation (our method) ──────────────────────────────

echo ">>> [3/7] Foveation 5-round, L0 start (our method)"
python eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/fov_r5_L0_crop0.3" \
    --rounds 5 --initial_level 0 --crop_ratio 0.3 \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

echo ">>> [4/7] Foveation 3-round, L0 start"
python eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/fov_r3_L0_crop0.3" \
    --rounds 3 --initial_level 0 --crop_ratio 0.3 \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

echo ">>> [5/7] Foveation 1-round, L1 (single-round original res)"
python eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/fov_r1_L1" \
    --rounds 1 --initial_level 1 \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

echo ">>> [6/7] Foveation 1-round, L2 (single-round high res)"
python eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/fov_r1_L2" \
    --rounds 1 --initial_level 2 \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

echo ">>> [7/7] Foveation 5-round, L1 start (ablation)"
python eval/eval_screenspot_pro_aligned.py \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA_PATH}" \
    --save_path "${SAVE_BASE}/${MODEL_NAME}/fov_r5_L1_crop0.3" \
    --rounds 5 --initial_level 1 --crop_ratio 0.3 \
    ${SAMPLE_FLAG} 2>&1 | tail -5
echo ""

echo "================================================================"
echo "  All evaluations complete. Results in:"
echo "  ${SAVE_BASE}/${MODEL_NAME}/"
echo "================================================================"
