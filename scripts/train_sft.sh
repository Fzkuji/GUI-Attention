#!/bin/bash
# ============================================================================
# Multi-Round Foveated SFT Training
# ============================================================================
# Teacher-forcing: crop around GT each round, supervise attention with NLL.
# Recommended as the first training stage before GRPO.
#
# Usage:
#   bash scripts/train_sft.sh
#   bash scripts/train_sft.sh 2   # override max_rounds
# ============================================================================

set -e

# ── Paths (modify these for your environment) ──────────────────────────────
BASE_DIR="/mnt/data/zichuanfu"
PROJECT_DIR="${BASE_DIR}/GUI-Attention"
GUI_AIMA_SRC="${BASE_DIR}/Experiments/GUI-AIMA/src"

MODEL="${BASE_DIR}/models/GUI-AIMA-3B"
DATA="${BASE_DIR}/data/GUI-Actor/guiact_bbox.json"
IMAGE_FOLDER="${BASE_DIR}/data/GUI-Actor/images/GUIAct/web_imgs"
OUTPUT_DIR="${BASE_DIR}/checkpoints/sft_multi_round"

# ── Hyperparameters ─────────────────────────────────────────────────────────
MAX_ROUNDS=${1:-2}          # SFT only needs 2 rounds (round 0 low-res + round 1 high-res)
CROP_RATIO=0.3
EPOCHS=3
BATCH_SIZE=1
GRAD_ACCUM=8
LR=5e-6
LOGGING_STEPS=10
SAVE_STEPS=500

# ── Run ─────────────────────────────────────────────────────────────────────
export PYTHONPATH="${GUI_AIMA_SRC}:${PYTHONPATH}"
cd "${PROJECT_DIR}"

echo "=== SFT Training ==="
echo "  model:      ${MODEL}"
echo "  data:       ${DATA}"
echo "  output:     ${OUTPUT_DIR}"
echo "  max_rounds: ${MAX_ROUNDS}"
echo "  lr:         ${LR}"
echo ""

python train_grpo_multi_round.py \
    --training_mode sft \
    --model_name_or_path "${MODEL}" \
    --data_path "${DATA}" \
    --image_folder "${IMAGE_FOLDER}" \
    --output_dir "${OUTPUT_DIR}" \
    --min_pixels 3136 \
    --max_pixels 1003520 \
    --max_rounds "${MAX_ROUNDS}" \
    --crop_ratio "${CROP_RATIO}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LR}" \
    --weight_decay 0.01 \
    --logging_steps "${LOGGING_STEPS}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none
