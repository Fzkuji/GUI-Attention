"""
GUI-Attention Training Script.

Trains the FoveatedGroundingHead (~579 params) on GTA 60K data.
Base Qwen2.5-VL model is frozen; only grounding head is trained.

Usage:
    python train.py \
        --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
        --data_path data/gta/gta_data_wo_web_output_60k.json \
        --images_folder /root/autodl-tmp/data/gta_images \
        [--synthetic_mode] \
        [--wandb]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent / "src"))

from gui_attention.model.foveated_qwen25vl import FoveatedQwen25VL
from gui_attention.foveation.sampler import FoveatedSampler
from gui_attention.data.dataset import GTADataset, GTACollator


def create_synthetic_dataset(save_path: str, num_samples: int = 500):
    """Create a synthetic GTA-format JSON for testing the pipeline."""
    import random
    random.seed(42)
    samples = []
    for i in range(num_samples):
        x = round(random.uniform(0.05, 0.95), 4)
        y = round(random.uniform(0.05, 0.95), 4)
        samples.append({
            "image": f"synthetic_{i}.png",
            "conversations": [
                {"from": "human", "value": f"<image> Click element {i}"},
                {"from": "gpt", "value": f"pyautogui.click(x={x}, y={y})",
                 "bbox_gt": [max(0, x-0.01), max(0, y-0.01), min(1, x+0.01), min(1, y+0.01)]}
            ]
        })
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(samples, f)
    print(f"Created synthetic dataset: {save_path} ({num_samples} samples)")
    return save_path


def train(args):
    print("=" * 60)
    print("GUI-Attention Training")
    print("=" * 60)

    # Setup wandb
    if args.wandb:
        try:
            import wandb
            wandb.init(project="gui-attention", config=vars(args))
        except ImportError:
            print("wandb not installed, disabling logging")
            args.wandb = False

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    t0 = time.time()
    model = FoveatedQwen25VL(model_name_or_path=args.model_path)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Print param counts
    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for _, p in trainable_params)
    print(f"\nTrainable parameters ({total_trainable} total):")
    for name, param in trainable_params:
        print(f"  {name}: {param.shape} ({param.numel()} params)")

    # Handle synthetic mode
    data_path = args.data_path
    if args.synthetic_mode or not os.path.exists(args.data_path):
        if not os.path.exists(args.data_path):
            print(f"\nData file not found: {args.data_path}")
            print("Falling back to synthetic mode...")
        syn_path = "data/synthetic_train.json"
        data_path = create_synthetic_dataset(syn_path, num_samples=args.synthetic_samples)
        args.synthetic_mode = True

    # Create dataset
    print(f"\nLoading dataset from {data_path}...")
    sampler = FoveatedSampler()
    dataset = GTADataset(
        data_path=data_path,
        images_folder=args.images_folder,
        processor=model.processor,
        sampler=sampler,
        fixation_noise_std=args.fixation_noise_std,
        synthetic_mode=args.synthetic_mode,
    )
    print(f"Dataset: {len(dataset)} samples")

    collator = GTACollator(pad_token_id=model.processor.tokenizer.pad_token_id or 0)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # Optimizer - only grounding head params
    optimizer = torch.optim.AdamW(
        [p for p in model.grounding_head.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR scheduler
    total_steps = len(dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.01)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total steps: {total_steps}")

    global_step = 0
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        # Keep base model in eval mode (frozen)
        model.model.eval()

        epoch_loss = 0.0
        epoch_dist = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move tensors to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                image_grid_thw = batch["image_grid_thw"].to(device)
                gt_coords = batch["gt_coords"].to(device)

                # Forward
                result = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    gt_coords=gt_coords,
                    crop_bboxes=batch["crop_bboxes"],
                    crop_levels=batch["crop_levels"],
                )

                loss = result["loss"]

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Metrics
                with torch.no_grad():
                    pred = result["pred_coords"]
                    gt = result["gt_coords"].to(device)
                    dist = torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean().item()

                epoch_loss += loss.item()
                epoch_dist += dist
                num_batches += 1
                global_step += 1

                # Log
                if global_step % args.log_every == 0:
                    avg_loss = epoch_loss / num_batches
                    avg_dist = epoch_dist / num_batches
                    lr = scheduler.get_last_lr()[0]
                    print(f"  [Epoch {epoch+1}/{args.epochs}] Step {global_step}: "
                          f"loss={loss.item():.4f} avg_loss={avg_loss:.4f} "
                          f"avg_dist={avg_dist:.4f} lr={lr:.2e}")

                    if args.wandb:
                        import wandb
                        wandb.log({
                            "loss": loss.item(),
                            "avg_loss": avg_loss,
                            "avg_dist": avg_dist,
                            "lr": lr,
                            "epoch": epoch,
                            "step": global_step,
                        })

            except Exception as e:
                print(f"  Error at batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Epoch summary
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_dist = epoch_dist / num_batches
            print(f"\nEpoch {epoch+1}/{args.epochs}: avg_loss={avg_loss:.4f} avg_dist={avg_dist:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(args.output_dir, "grounding_head_best.pt")
                torch.save(model.grounding_head.state_dict(), save_path)
                print(f"  Saved best model to {save_path}")

    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "grounding_head_final.pt")
    torch.save(model.grounding_head.state_dict(), final_path)
    print(f"\nSaved final model to {final_path}")

    # Also save full config for reproducibility
    config_path = os.path.join(args.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to {config_path}")

    if args.wandb:
        import wandb
        wandb.finish()

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train GUI-Attention grounding head")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Qwen2.5-VL checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to GTA JSON data file")
    parser.add_argument("--images_folder", type=str, default="",
                        help="Root folder for images")
    parser.add_argument("--output_dir", type=str, default="checkpoints/gui-attention",
                        help="Output directory for checkpoints")
    parser.add_argument("--synthetic_mode", action="store_true",
                        help="Use synthetic images when real ones not found")
    parser.add_argument("--synthetic_samples", type=int, default=500,
                        help="Number of synthetic samples to generate")

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (1 recommended due to variable-length sequences)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (high because only ~579 params)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fixation_noise_std", type=float, default=0.05)

    # Logging
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
