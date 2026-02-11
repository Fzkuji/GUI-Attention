"""
Train Qwen2.5-VL with pointer-token attention supervision for GUI grounding.

Adds special pointer tokens and trains the model with:
1. LM loss: predict correct text (pyautogui.click(<pointer_start><pointer_pad><pointer_end>))
2. Pointer loss: KL-div supervision forcing pointer_pad attention onto bbox visual patches

Based on GUI-AIMA's training approach.
"""

import json
import glob
import os
import random
import math

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5_VLProcessor,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

from gui_attention.model.foveated_qwen25vl import (
    Qwen25VLWithPointer,
    POINTER_START_TOKEN,
    POINTER_PAD_TOKEN,
    POINTER_END_TOKEN,
    IMAGE_PAD_ID,
)


def setup_pointer_tokens(model, processor):
    """Add pointer tokens to tokenizer and model."""
    special_tokens = [POINTER_START_TOKEN, POINTER_PAD_TOKEN, POINTER_END_TOKEN]
    num_added = processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    print(f"Added {num_added} special tokens")

    # Resize embeddings
    model.resize_token_embeddings(len(processor.tokenizer))

    # Store token IDs in config
    model.config.pointer_start_token_id = processor.tokenizer.convert_tokens_to_ids(POINTER_START_TOKEN)
    model.config.pointer_pad_token_id = processor.tokenizer.convert_tokens_to_ids(POINTER_PAD_TOKEN)
    model.config.pointer_end_token_id = processor.tokenizer.convert_tokens_to_ids(POINTER_END_TOKEN)

    print(f"Pointer token IDs: start={model.config.pointer_start_token_id}, "
          f"pad={model.config.pointer_pad_token_id}, end={model.config.pointer_end_token_id}")
    return model, processor


class ScreenSpotProPointerDataset(Dataset):
    """ScreenSpot-Pro dataset with pointer tokens and attention labels."""

    def __init__(self, data_dir, processor, model_config, max_samples=None, max_image_size=800):
        self.data_dir = data_dir
        self.processor = processor
        self.model_config = model_config
        self.max_image_size = max_image_size
        self.samples = []

        ann_files = sorted(glob.glob(os.path.join(data_dir, "annotations", "*.json")))
        for f in ann_files:
            with open(f) as fp:
                self.samples.extend(json.load(fp))

        random.seed(42)
        random.shuffle(self.samples)
        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = os.path.join(self.data_dir, "images", s["img_filename"])
        img = Image.open(img_path).convert("RGB")

        # Resize
        w_orig, h_orig = img.size
        if max(w_orig, h_orig) > self.max_image_size:
            scale = self.max_image_size / max(w_orig, h_orig)
            img = img.resize((int(w_orig * scale), int(h_orig * scale)), Image.LANCZOS)

        # Compute normalized bbox
        bbox = s["bbox"]
        x1n, y1n = bbox[0] / w_orig, bbox[1] / h_orig
        x2n, y2n = bbox[2] / w_orig, bbox[3] / h_orig

        return {
            "image": img,
            "instruction": s["instruction"],
            "bbox_norm": (x1n, y1n, x2n, y2n),  # normalized bbox
        }


def collate_fn(batch, processor, model_config):
    """Build batch with pointer tokens and compute attention supervision labels."""
    texts = []
    images = []
    all_labels_list = []

    pointer_start_id = model_config.pointer_start_token_id
    pointer_pad_id = model_config.pointer_pad_token_id
    pointer_end_id = model_config.pointer_end_token_id

    for item in batch:
        img = item["image"]
        images.append(img)

        # Build conversation with pointer tokens in assistant response
        target_text = f"pyautogui.click({POINTER_START_TOKEN}{POINTER_PAD_TOKEN}{POINTER_END_TOKEN})"
        conversation = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": item["instruction"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": target_text},
            ]},
        ]
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        texts.append(text)

    # Process images
    image_inputs, _ = process_vision_info([
        [{"role": "user", "content": [{"type": "image", "image": img}]}]
        for img in images
    ])

    inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt")

    # Create LM labels (mask user input, keep assistant response)
    labels = inputs["input_ids"].clone()
    im_start_id = 151644  # <|im_start|>
    for i in range(len(batch)):
        ids = inputs["input_ids"][i].tolist()
        positions = [j for j, t in enumerate(ids) if t == im_start_id]
        if len(positions) >= 2:
            # Find newline after assistant header
            for j in range(positions[1], len(ids)):
                if ids[j] == 198:  # newline
                    mask_end = j + 1
                    break
            else:
                mask_end = positions[1] + 3
            labels[i, :mask_end] = -100
        labels[i, labels[i] == processor.tokenizer.pad_token_id] = -100

    # Compute multi-patch attention labels
    # For each sample, create a binary mask over visual tokens indicating which ones
    # fall within the target bbox
    merge_size = model_config.vision_config.spatial_merge_size
    multi_patch_labels = []

    img_idx = 0
    for i, item in enumerate(batch):
        x1n, y1n, x2n, y2n = item["bbox_norm"]
        input_ids = inputs["input_ids"][i]

        # Count visual tokens for this sample
        visual_mask = (input_ids == IMAGE_PAD_ID)
        n_visual = visual_mask.sum().item()

        # Get image grid for this sample
        if "image_grid_thw" in inputs and img_idx < inputs["image_grid_thw"].shape[0]:
            t, h, w = inputs["image_grid_thw"][img_idx].tolist()
            t, h, w = int(t), int(h), int(w)
            h_m, w_m = h // merge_size, w // merge_size

            # Create label: which visual tokens are in bbox
            label = torch.zeros(1, n_visual)
            token_idx = 0
            for _t in range(t):
                for row in range(h_m):
                    for col in range(w_m):
                        if token_idx >= n_visual:
                            break
                        # Token center in normalized coords
                        cx = (col + 0.5) / w_m
                        cy = (row + 0.5) / h_m
                        # Check if in bbox
                        if x1n <= cx <= x2n and y1n <= cy <= y2n:
                            label[0, token_idx] = 1.0
                        token_idx += 1

            # If no tokens in bbox (bbox too small), mark nearest token
            if label.sum() == 0:
                best_dist = float('inf')
                best_idx = 0
                cx_target = (x1n + x2n) / 2
                cy_target = (y1n + y2n) / 2
                token_idx = 0
                for _t in range(t):
                    for row in range(h_m):
                        for col in range(w_m):
                            if token_idx >= n_visual:
                                break
                            cx = (col + 0.5) / w_m
                            cy = (row + 0.5) / h_m
                            d = (cx - cx_target)**2 + (cy - cy_target)**2
                            if d < best_dist:
                                best_dist = d
                                best_idx = token_idx
                            token_idx += 1
                label[0, best_idx] = 1.0

            multi_patch_labels.append(label)
            img_idx += 1
        else:
            multi_patch_labels.append(torch.zeros(1, n_visual))

    inputs["labels"] = labels
    inputs["multi_patch_labels"] = multi_patch_labels
    return inputs


class PointerTrainer:
    """Custom training loop for pointer-token attention training."""

    def __init__(self, model, processor, train_dataset, args):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.args = args

        self.dataloader = DataLoader(
            train_dataset,
            batch_size=args["batch_size"],
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, processor, model.config),
            num_workers=2,
            pin_memory=True,
        )

    def train(self):
        device = next(self.model.parameters()).device
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args["lr"],
            weight_decay=0.01,
        )
        total_steps = len(self.dataloader) * self.args["epochs"] // self.args["grad_accum"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        step = 0
        for epoch in range(self.args["epochs"]):
            self.model.train()
            epoch_loss = 0
            epoch_lm_loss = 0
            epoch_ptr_loss = 0

            for batch_idx, batch in enumerate(self.dataloader):
                # Move to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                image_grid_thw = batch["image_grid_thw"].to(device)
                labels = batch["labels"].to(device)
                multi_patch_labels = [l.to(device) for l in batch["multi_patch_labels"]]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels,
                    multi_patch_labels=multi_patch_labels,
                    return_dict=True,
                )

                loss = outputs.loss / self.args["grad_accum"]
                loss.backward()

                if (batch_idx + 1) % self.args["grad_accum"] == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1

                epoch_loss += outputs.loss.item()
                if outputs.lm_loss is not None:
                    epoch_lm_loss += outputs.lm_loss.item()
                if outputs.pointer_loss is not None:
                    epoch_ptr_loss += outputs.pointer_loss.item()

                if (batch_idx + 1) % 10 == 0:
                    n = batch_idx + 1
                    print(f"  Epoch {epoch+1} [{n}/{len(self.dataloader)}] "
                          f"loss={epoch_loss/n:.4f} lm={epoch_lm_loss/n:.4f} ptr={epoch_ptr_loss/n:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")

            n = len(self.dataloader)
            print(f"Epoch {epoch+1}/{self.args['epochs']}: "
                  f"loss={epoch_loss/n:.4f} lm={epoch_lm_loss/n:.4f} ptr={epoch_ptr_loss/n:.4f}")

            # Save checkpoint
            save_dir = os.path.join(self.args["output_dir"], f"checkpoint-epoch{epoch+1}")
            self.model.save_pretrained(save_dir)
            self.processor.save_pretrained(save_dir)
            print(f"Saved checkpoint to {save_dir}")

        # Save final
        save_dir = os.path.join(self.args["output_dir"], "final")
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        print(f"Final model saved to {save_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data_dir", default="/root/autodl-tmp/data/ScreenSpot-Pro")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/checkpoints/gui-attention-pointer")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_image_size", type=int, default=800)
    parser.add_argument("--pointer_loss_weight", type=float, default=1.0)
    parser.add_argument("--lm_loss_weight", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = Qwen25VLWithPointer.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(args.model_path)

    # Add pointer tokens
    model, processor = setup_pointer_tokens(model, processor)

    # Set loss weights
    model.pointer_loss_weight = args.pointer_loss_weight
    model.lm_loss_weight = args.lm_loss_weight

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to("cuda")

    # Dataset
    dataset = ScreenSpotProPointerDataset(
        args.data_dir, processor, model.config,
        max_samples=args.max_samples, max_image_size=args.max_image_size,
    )

    # Train
    trainer = PointerTrainer(
        model=model, processor=processor, train_dataset=dataset,
        args={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "output_dir": args.output_dir,
        },
    )
    trainer.train()


if __name__ == "__main__":
    main()
