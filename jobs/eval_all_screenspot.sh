#!/bin/bash
# Evaluate on all 3 ScreenSpot benchmarks (Pro, v1, v2)
# Uses the unified eval/eval_screenspot.py script with DDP.
#
# Usage:
#   CHECKPOINT=/path/to/ckpt NUM_GPUS=8 bash jobs/eval_all_screenspot.sh
#   CHECKPOINT=/path/to/ckpt BASE_MODEL=/path/to/model bash jobs/eval_all_screenspot.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

CHECKPOINT=${CHECKPOINT:?"Set CHECKPOINT=/path/to/checkpoint"}
BASE_MODEL=${BASE_MODEL:-"/mnt/data/zichuanfu/GUI-Attention-Workspace/models/GUI-Actor-3B-Qwen2.5-VL"}
NUM_GPUS=${NUM_GPUS:-8}
ROUNDS=${ROUNDS:-6}
DATA_DIR=${DATA_DIR:-"/mnt/data/zichuanfu/GUI-Attention-Workspace/data"}

echo "=== Evaluating checkpoint: $CHECKPOINT ==="
echo "  base_model=$BASE_MODEL"
echo "  rounds=$ROUNDS  num_gpus=$NUM_GPUS"
echo "  adaptive_crop=OFF (fixed crop_size=308, crop_upscale=3)"
echo ""

# 1. ScreenSpot-Pro
echo ">>> ScreenSpot-Pro"
torchrun --nproc_per_node=$NUM_GPUS eval/eval_screenspot.py \
    --dataset pro \
    --data_dir "$DATA_DIR/ScreenSpot-Pro" \
    --checkpoint "$CHECKPOINT" \
    --base_model "$BASE_MODEL" \
    --rounds "$ROUNDS" \
    --use_dual_tokens \
    --no_adaptive_crop
echo ""

# 2. ScreenSpot v1 (downloads from HuggingFace)
echo ">>> ScreenSpot v1"
torchrun --nproc_per_node=$NUM_GPUS eval/eval_screenspot.py \
    --dataset v1 \
    --checkpoint "$CHECKPOINT" \
    --base_model "$BASE_MODEL" \
    --rounds "$ROUNDS" \
    --use_dual_tokens \
    --no_adaptive_crop
echo ""

# 3. ScreenSpot v2
echo ">>> ScreenSpot v2"
torchrun --nproc_per_node=$NUM_GPUS eval/eval_screenspot.py \
    --dataset v2 \
    --data_dir "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" \
    --base_model "$BASE_MODEL" \
    --rounds "$ROUNDS" \
    --use_dual_tokens \
    --no_adaptive_crop
echo ""

echo "=== All evaluations done ==="
