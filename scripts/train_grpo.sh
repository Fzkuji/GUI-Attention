#!/bin/bash
# ============================================================================
# Multi-Round Foveated GRPO Training
# ============================================================================
# Policy gradient: sample N trajectories, reinforce the better ones.
# Recommended as the second stage after SFT warm-up.
#
# Usage:
#   bash scripts/train_grpo.sh
#   bash scripts/train_grpo.sh /path/to/sft/checkpoint   # start from SFT checkpoint
# ============================================================================

set -e

# ── Paths (modify these for your environment) ──────────────────────────────
BASE_DIR="/mnt/data/zichuanfu"
PROJECT_DIR="${BASE_DIR}/GUI-Attention"
GUI_AIMA_SRC="${BASE_DIR}/Experiments/GUI-AIMA/src"

# Use SFT checkpoint if provided, otherwise use base model
MODEL="${1:-${BASE_DIR}/models/GUI-AIMA-3B}"
DATA="${BASE_DIR}/data/GUI-Actor/guiact_bbox.json"
IMAGE_FOLDER="${BASE_DIR}/data/GUI-Actor/images/GUIAct/web_imgs"
OUTPUT_DIR="${BASE_DIR}/checkpoints/grpo_multi_round"

# ── Hyperparameters ─────────────────────────────────────────────────────────
MAX_ROUNDS=3
CROP_RATIO=0.3
NUM_GENERATIONS=4           # more generations = more stable advantage, but slower
EPOCHS=2
BATCH_SIZE=1
GRAD_ACCUM=8
LR=1e-6                    # lower LR than SFT (policy gradient is noisier)
TEMPERATURE=0.9
ATTN_TEMPERATURE=1.0
LOGGING_STEPS=10
SAVE_STEPS=500

# ── Run ─────────────────────────────────────────────────────────────────────
export PYTHONPATH="${GUI_AIMA_SRC}:${PYTHONPATH}"
cd "${PROJECT_DIR}"

echo "=== GRPO Training ==="
echo "  model:           ${MODEL}"
echo "  data:            ${DATA}"
echo "  output:          ${OUTPUT_DIR}"
echo "  max_rounds:      ${MAX_ROUNDS}"
echo "  num_generations: ${NUM_GENERATIONS}"
echo "  lr:              ${LR}"
echo ""

python train_grpo_multi_round.py \
    --training_mode grpo \
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
    --num_generations "${NUM_GENERATIONS}" \
    --max_completion_length 256 \
    --temperature "${TEMPERATURE}" \
    --attn_temperature "${ATTN_TEMPERATURE}" \
    --beta 0.0 \
    --epsilon 0.2 \
    --learning_rate "${LR}" \
    --weight_decay 0.01 \
    --logging_steps "${LOGGING_STEPS}" \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none
