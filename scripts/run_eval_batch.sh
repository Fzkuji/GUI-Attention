#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
cd /root/GUI-Attention

MAX_SAMPLES=${1:-50}
QWEN_MODEL="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
AIMA_MODEL="/root/autodl-tmp/models/GUI-AIMA-3B"

echo "=== Batch eval with max_samples=$MAX_SAMPLES ==="

# Test 1: Qwen2.5-VL-3B, single round, high res
echo ">>> [1/6] Qwen2.5-VL-3B, single, high"
python eval/eval_screenspot_pro.py --model_name_or_path $QWEN_MODEL --rounds 1 --resolution high --max_samples $MAX_SAMPLES 2>&1 | tail -20

# Test 2: GUI-AIMA-3B, single round, high res
echo ">>> [2/6] GUI-AIMA-3B, single, high"
python eval/eval_screenspot_pro.py --model_name_or_path $AIMA_MODEL --rounds 1 --resolution high --max_samples $MAX_SAMPLES 2>&1 | tail -20

# Test 3: GUI-AIMA-3B, single round, low res
echo ">>> [3/6] GUI-AIMA-3B, single, low"
python eval/eval_screenspot_pro.py --model_name_or_path $AIMA_MODEL --rounds 1 --resolution low --max_samples $MAX_SAMPLES 2>&1 | tail -20

# Test 4: GUI-AIMA-3B, 2 rounds, crop_resize, high res
echo ">>> [4/6] GUI-AIMA-3B, 2-round crop_resize, high"
python eval/eval_screenspot_pro.py --model_name_or_path $AIMA_MODEL --rounds 2 --multi_strategy crop_resize --resolution high --max_samples $MAX_SAMPLES 2>&1 | tail -20

# Test 5: GUI-AIMA-3B, 3 rounds, crop_resize, high res
echo ">>> [5/6] GUI-AIMA-3B, 3-round crop_resize, high"
python eval/eval_screenspot_pro.py --model_name_or_path $AIMA_MODEL --rounds 3 --multi_strategy crop_resize --resolution high --max_samples $MAX_SAMPLES 2>&1 | tail -20

# Test 6: GUI-AIMA-3B, 5 rounds, token_concat, high res
echo ">>> [6/6] GUI-AIMA-3B, 5-round token_concat, high"
python eval/eval_screenspot_pro.py --model_name_or_path $AIMA_MODEL --rounds 5 --multi_strategy token_concat --resolution high --max_samples $MAX_SAMPLES 2>&1 | tail -20

echo "=== All tests done ==="
