#!/bin/bash
# GRPO Training with Attention Sampling
# Run on server: bash scripts/train_grpo_attention.sh

export PYTHONPATH="/root/GUI-AIMA/src:${PYTHONPATH}"

python train_grpo_attention.py \
    --model_name_or_path /root/autodl-tmp/models/GUI-AIMA-3B \
    --data_path /root/autodl-tmp/data/GUI-Actor/guiact_bbox.json \
    --image_folder /root/autodl-tmp/data/GUI-Actor/images/GUIAct/web_imgs \
    --output_dir /root/autodl-tmp/checkpoints/grpo_attn_sampling \
    --format_reward_value 0.05 \
    --use_placeholder true \
    --min_pixels 3136 \
    --max_pixels 1003520 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_generations 8 \
    --max_completion_length 256 \
    --temperature 0.9 \
    --attn_temperature 1.0 \
    --beta 0.0 \
    --epsilon 0.2 \
    --learning_rate 1e-6 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --report_to none
