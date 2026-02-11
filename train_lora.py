"""
LoRA fine-tune Qwen2.5-VL-3B for GUI grounding.

Train the model to output click coordinates given a screenshot + instruction.
This teaches the model's attention to focus on target UI elements.
"""

import json
import glob
import os
import random
from dataclasses import dataclass, field

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info


class ScreenSpotProDataset(Dataset):
    """ScreenSpot-Pro dataset for grounding training."""

    def __init__(self, data_dir, processor, max_samples=None, split="train", train_ratio=0.9):
        self.data_dir = data_dir
        self.processor = processor
        self.samples = []

        # Load all annotation files
        ann_files = sorted(glob.glob(os.path.join(data_dir, "annotations", "*.json")))
        for f in ann_files:
            with open(f) as fp:
                data = json.load(fp)
                self.samples.extend(data)

        # Shuffle deterministically and split
        random.seed(42)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * train_ratio)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} samples for {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.data_dir, "images", s["img_filename"])

        # Compute normalized center coords from bbox
        bbox = s["bbox"]
        w, h = s["img_size"]
        cx = (bbox[0] + bbox[2]) / (2 * w)
        cy = (bbox[1] + bbox[3]) / (2 * h)

        return {
            "image_path": img_path,
            "instruction": s["instruction"],
            "target_x": cx,
            "target_y": cy,
        }


def collate_fn(batch, processor):
    """Collate batch with Qwen2.5-VL processor."""
    texts = []
    images = []

    for item in batch:
        img = Image.open(item["image_path"]).convert("RGB")
        # Resize to reduce memory (ScreenSpot-Pro images are huge 3840x2160)
        max_size = 1280
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        images.append(img)

        # Build conversation
        target_text = f"pyautogui.click(x={item['target_x']:.4f}, y={item['target_y']:.4f})"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": item["instruction"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text},
                ],
            },
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # Process with vision
    image_inputs, _ = process_vision_info([
        [{"role": "user", "content": [{"type": "image", "image": img}]}]
        for img in images
    ])

    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Create labels: mask everything except the assistant response
    labels = inputs["input_ids"].clone()
    # Find assistant token positions and mask everything before them
    for i in range(len(batch)):
        # Find the assistant header position
        ids = inputs["input_ids"][i].tolist()
        # The assistant response starts after "<|im_start|>assistant\n"
        # We want to mask all tokens before the response content
        # Simple approach: find the last occurrence of "assistant" marker
        # and unmask only tokens after it
        # Token for <|im_start|> is 151644
        im_start_id = 151644
        positions = [j for j, t in enumerate(ids) if t == im_start_id]
        if len(positions) >= 2:
            # Second <|im_start|> is the assistant header
            assistant_start = positions[1]
            # Find the newline after "assistant"
            for j in range(assistant_start, len(ids)):
                if ids[j] == 198:  # newline token
                    mask_end = j + 1
                    break
            else:
                mask_end = assistant_start + 3  # fallback
            labels[i, :mask_end] = -100
        # Also mask padding
        labels[i, labels[i] == processor.tokenizer.pad_token_id] = -100

    inputs["labels"] = labels
    return inputs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/data/ScreenSpot-Pro")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/checkpoints/gui-attention-lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    print("Loading model and processor...")
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Need eager for attention extraction later
    )

    # LoRA config — target attention layers
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    train_dataset = ScreenSpotProDataset(
        args.data_dir, processor, max_samples=args.max_samples, split="train"
    )
    eval_dataset = ScreenSpotProDataset(
        args.data_dir, processor, max_samples=min(50, args.max_samples or 50), split="eval"
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=2,
        gradient_checkpointing=True,
    )

    # Custom collate
    def custom_collate(batch):
        return collate_fn(batch, processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_collate,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Save final LoRA weights
    model.save_pretrained(os.path.join(args.output_dir, "final"))
    print(f"LoRA weights saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
