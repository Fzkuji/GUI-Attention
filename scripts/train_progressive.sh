#!/bin/bash
# Progressive Resolution Training Script
# Run on server with: bash scripts/train_progressive.sh

export PYTHONPATH="/root/GUI-AIMA/src:${PYTHONPATH}"

python train_progressive.py \
    --model_name_or_path /root/autodl-tmp/models/GUI-AIMA-3B \
    --data_path /root/autodl-tmp/data/GUI-Actor/guiact_bbox.json \
    --image_folder /root/autodl-tmp/data/GUI-Actor/images/GUIAct/web_imgs \
    --output_dir /root/autodl-tmp/checkpoints/progressive_full \
    --max_pixels 1003520 \
    --crop_pixels 501760 \
    --min_pixels 3136 \
    --crop_ratio 0.3 \
    --crop_jitter 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 2000 \
    --save_total_limit 3 \
    --bf16 true \
    --gradient_checkpointing true \
    --pointer_loss_weight 1.0 \
    --lm_loss_weight 1.0 \
    --weighting query_1 \
    --dataloader_num_workers 4 \
    --report_to none
