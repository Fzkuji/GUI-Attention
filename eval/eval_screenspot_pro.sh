#!/bin/bash
# ScreenSpot-Pro Evaluation Script
# Usage: bash eval/eval_screenspot_pro.sh [--trained /path/to/grounding_head.pt]

set -e

MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
DATA_PATH="/root/autodl-tmp/data/ScreenSpot-Pro"
SAVE_PATH="/root/autodl-tmp/results/screenspot_pro"

# Baseline (untrained grounding head)
echo "=== Running baseline evaluation ==="
python eval/eval_screenspot_pro.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH" \
    --fixation center

# If trained grounding head path provided, run trained eval too
if [ "$1" == "--trained" ] && [ -n "$2" ]; then
    echo ""
    echo "=== Running trained model evaluation ==="
    python eval/eval_screenspot_pro.py \
        --model_path "$MODEL_PATH" \
        --data_path "$DATA_PATH" \
        --save_path "$SAVE_PATH" \
        --grounding_head_path "$2" \
        --fixation center
fi

echo ""
echo "Done! Results saved to $SAVE_PATH"
