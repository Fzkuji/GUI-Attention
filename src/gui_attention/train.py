"""
Saccade Foveation Training (v4): LoRA + ActionHead, natural multi-round saccade.

No gui_aima dependency. Uses transformers Qwen2.5-VL + peft LoRA + ActionHead.

Each sample trains up to max_saccade_rounds, using the model's OWN predictions
to decide where to crop (not teacher forcing). Three scenarios per round:

  Round 0: [low-res full image] [instruction] [anchor_0]
    → forward → action head → loss (binary overlap with GT)
    → model predicts a point → used as crop center for round 1

  Round 1+ (GT in crop):
    → labels on high-res crop patches (local GT overlap)
    → low-res patches covered by crops are masked
    → stop (target found)

  Round 1+ (GT NOT in crop — saccade):
    → labels on UNMASKED low-res patches overlapping GT
    → high-res patches get 0 labels
    → model predicts new point → next crop center
    → continue loop

  Total loss = mean of all valid round losses

Usage:
    python -m gui_attention.train \
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --data_path /path/to/guiact_bbox.json \
        --image_folder /path/to/images \
        --output_dir /path/to/checkpoints
"""

import json
import math
import os
import random
import re
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import transformers
from PIL import Image, ImageFile
from tqdm import tqdm

from gui_attention.attention import (
    extract_anchor_hidden_states,
    extract_visual_hidden_states,
    identify_attended_image,
    token_to_spatial,
)
from gui_attention.builder import MultiRoundInputBuilder
from gui_attention.constants import HIGH_RES_MAX_PIXELS, LOW_RES_MAX_PIXELS
from gui_attention.crop import crop_image, point_in_bbox
from gui_attention.labels import compute_binary_labels, compute_overlap_mask
from gui_attention.model import Qwen25VLWithActionHead, build_model

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -- Arguments ----------------------------------------------------------------

@dataclass
class ScriptArgs:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    data_path: str = field(default=None, metadata={"help": "JSON path(s), comma-separated for multiple datasets"})
    image_folder: str = field(default=None, metadata={"help": "Image folder(s), comma-separated, matching data_path order"})
    max_samples: Optional[int] = field(default=None)
    min_pixels: int = field(default=3136)
    low_res_max_pixels: int = field(default=LOW_RES_MAX_PIXELS)
    high_res_max_pixels: int = field(default=HIGH_RES_MAX_PIXELS)
    crop_ratio: float = field(default=0.3)
    crop_jitter: float = field(default=0.05, metadata={"help": "Random jitter for crop center (fraction of image)"})
    max_saccade_rounds: int = field(default=3, metadata={"help": "Max rounds per sample (round 0 + up to N-1 saccades)"})
    # LoRA
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_target_modules: str = field(default="q_proj,v_proj")
    # Dual LR
    action_head_lr: float = field(default=1e-4)
    lora_lr: float = field(default=5e-5)


@dataclass
class TrainArgs(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    gradient_checkpointing: bool = field(default=True)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)


# -- Data ---------------------------------------------------------------------

def load_single_dataset(data_path, image_folder):
    with open(data_path) as f:
        raw = json.load(f)
    samples = []
    for item in raw:
        img_file = item["image"]
        if isinstance(img_file, list):
            img_file = img_file[0]
        img_path = os.path.join(image_folder, img_file)
        if not os.path.exists(img_path):
            continue
        bbox_gt = None
        user_text = ""
        for conv in item["conversations"]:
            if conv.get("bbox_gt") is not None:
                bbox_gt = conv["bbox_gt"]
            role = conv.get("from", conv.get("role", ""))
            if role in ("human", "user"):
                user_text = re.sub(r"<image>", "", conv.get("value", conv.get("content", ""))).strip()
        if bbox_gt and user_text:
            samples.append({"image_path": img_path, "instruction": user_text, "bbox_gt": bbox_gt})
    return samples


def load_dataset(data_path, image_folder, max_samples=None):
    """Load one or more datasets. Comma-separated paths for multiple datasets."""
    data_paths = [p.strip() for p in data_path.split(",")]
    image_folders = [p.strip() for p in image_folder.split(",")]
    if len(image_folders) == 1 and len(data_paths) > 1:
        image_folders = image_folders * len(data_paths)
    assert len(data_paths) == len(image_folders), \
        f"Mismatch: {len(data_paths)} data_paths vs {len(image_folders)} image_folders"

    samples = []
    for dp, imf in zip(data_paths, image_folders):
        s = load_single_dataset(dp, imf)
        print(f"  {os.path.basename(dp)}: {len(s)} samples")
        samples.extend(s)

    if max_samples:
        random.shuffle(samples)
        samples = samples[:max_samples]
    print(f"Total: {len(samples)} samples")
    return samples


# -- Helpers ------------------------------------------------------------------

def _get_model_device(model):
    if hasattr(model, 'device'):
        return model.device
    return next(model.parameters()).device


# -- Trainer ------------------------------------------------------------------

class SaccadeTrainer:
    def __init__(self, model: Qwen25VLWithActionHead, tokenizer, train_data,
                 args: TrainArgs, script_args: ScriptArgs):
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.args = args
        self.sa = script_args
        self.use_deepspeed = HAS_DEEPSPEED and args.deepspeed is not None

        self.builder = MultiRoundInputBuilder(
            script_args.model_name_or_path, tokenizer, script_args.min_pixels,
            low_res_max_pixels=script_args.low_res_max_pixels,
            high_res_max_pixels=script_args.high_res_max_pixels,
        )

        # Determine world_size early for total_steps calculation
        ws = dist.get_world_size() if dist.is_initialized() else 1
        total_steps = (
            int(args.num_train_epochs) * len(train_data)
            // ws
            // max(args.per_device_train_batch_size, 1)
            // max(args.gradient_accumulation_steps, 1)
        )
        if args.max_steps and args.max_steps > 0:
            total_steps = min(total_steps, args.max_steps)

        if self.use_deepspeed:
            self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=model,
                model_parameters=[p for p in model.parameters() if p.requires_grad],
                config=args.deepspeed,
            )
            self.model = self.model_engine.module
            self.rank = dist.get_rank() if dist.is_initialized() else 0
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        else:
            self.model = model
            self.model_engine = None
            # Detect distributed (torchrun)
            if dist.is_initialized():
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            else:
                self.rank = 0
                self.world_size = 1
            # Dual LR: action_head gets higher LR, LoRA gets lower LR
            action_head_params = list(model.action_head.parameters())
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            param_groups = [
                {"params": action_head_params, "lr": script_args.action_head_lr},
                {"params": backbone_params, "lr": script_args.lora_lr},
            ]
            self.optimizer = torch.optim.AdamW(
                param_groups, weight_decay=args.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7,
            )


        self.metrics = defaultdict(list)
        self.global_step = 0

    # -- multi-round saccade forward ----------------------------------------

    def _forward_sample(self, sample):
        """Multi-round saccade with model's own predictions for crop locations.

        Round 0: low-res → predict → loss
        Round 1+:
          - Crop at MODEL's prediction (not GT)
          - GT in crop → supervise high-res localization, stop
          - GT not in crop → supervise saccade (unmasked low-res GT patches)

        Returns (total_loss, num_valid_rounds, round_preds).
        """
        device = _get_model_device(self.model)
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return torch.tensor(0.0, device=device, requires_grad=True), 0, []

        bbox_gt = sample["bbox_gt"]
        gt_cx = (bbox_gt[0] + bbox_gt[2]) / 2
        gt_cy = (bbox_gt[1] + bbox_gt[3]) / 2

        img_tok = self.model.config.image_token_id
        pp_id = self.model.config.pointer_pad_token_id
        merge = self.model.backbone.base_model.model.visual.spatial_merge_size

        self.builder.reset()
        self.model.train()
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_valid = 0
        round_preds = []

        # ===== Round 0: low-res full image =====
        r0_inputs, cur_text, cur_images = self.builder.build_round0(
            sample, sample["instruction"],
        )
        inp0 = {k: v.to(device) for k, v in r0_inputs.items()}

        outputs0 = self.model(
            input_ids=inp0["input_ids"],
            attention_mask=inp0.get("attention_mask"),
            pixel_values=inp0.get("pixel_values"),
            image_grid_thw=inp0.get("image_grid_thw"),
        )
        last_hs0 = outputs0.hidden_states[-1]

        vis_hidden0, vis_ranges0 = extract_visual_hidden_states(
            last_hs0, inp0["input_ids"], img_tok)
        anchor0 = extract_anchor_hidden_states(
            last_hs0, inp0["input_ids"], pp_id, n=0)

        if vis_hidden0 is None or anchor0 is None:
            return total_loss, 0, []

        grid_dims0 = self.builder.get_image_grid_dims(inp0["image_grid_thw"], merge)
        nh0, nw0 = grid_dims0[0]

        labels0 = compute_binary_labels(nh0, nw0, bbox_gt).to(device).unsqueeze(0)
        attn0, loss0 = self.model.action_head(vis_hidden0, anchor0, labels=labels0)

        if loss0 is not None:
            total_loss = total_loss + loss0
            n_valid += 1

        # Model's round-0 prediction (used as crop center for round 1)
        with torch.no_grad():
            _, local_idx0 = identify_attended_image(attn0.squeeze(0), vis_ranges0)
            pred_x, pred_y = token_to_spatial(local_idx0, nw0, nh0)
            round_preds.append((pred_x, pred_y))

        # ===== Subsequent rounds: saccade with model's own predictions =====
        # LLM sees full history (all crops accumulated). Action head only
        # considers unmasked low-res + latest crop; old crops are masked out.
        for round_n in range(1, self.sa.max_saccade_rounds):
            # Crop at model's prediction from previous round
            cropped, crop_bbox = crop_image(img, pred_x, pred_y, self.sa.crop_ratio)

            # Check if GT center is findable in this crop
            gt_in_crop = point_in_bbox(gt_cx, gt_cy, crop_bbox)

            # Extend context (accumulate all crops for LLM context)
            try:
                r_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                    cur_text, cur_images, cropped, crop_bbox,
                )
            except Exception:
                break

            inp = {k: v.to(device) for k, v in r_inputs.items()}

            outputs = self.model(
                input_ids=inp["input_ids"],
                attention_mask=inp.get("attention_mask"),
                pixel_values=inp.get("pixel_values"),
                image_grid_thw=inp.get("image_grid_thw"),
            )
            last_hs = outputs.hidden_states[-1]

            vis_hidden, vis_ranges = extract_visual_hidden_states(
                last_hs, inp["input_ids"], img_tok)
            anchor = extract_anchor_hidden_states(
                last_hs, inp["input_ids"], pp_id, n=round_n)

            if vis_hidden is None or anchor is None or len(vis_ranges) < 2:
                break

            grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)
            n_low = vis_ranges[0][1]
            n_total = vis_hidden.shape[0]
            latest_img_idx = len(vis_ranges) - 1

            # Action head mask: current crop masks low-res + ALL old crops masked out
            this_crop_mask = compute_overlap_mask(nh0, nw0, crop_bbox)
            full_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
            full_mask[:n_low] = this_crop_mask.to(device)
            # Mask all previous crop tokens (indices 1 to latest-1)
            for prev_i in range(1, latest_img_idx):
                off, ntok = vis_ranges[prev_i]
                full_mask[off:off + ntok] = True

            if gt_in_crop:
                # --- Target found in crop: supervise high-res localization ---
                nh_high, nw_high = grid_dims[latest_img_idx]

                cbx1, cby1, cbx2, cby2 = crop_bbox
                cbw, cbh = cbx2 - cbx1, cby2 - cby1
                if cbw > 0 and cbh > 0:
                    local_gt = (
                        max(0.0, min(1.0, (bbox_gt[0] - cbx1) / cbw)),
                        max(0.0, min(1.0, (bbox_gt[1] - cby1) / cbh)),
                        max(0.0, min(1.0, (bbox_gt[2] - cbx1) / cbw)),
                        max(0.0, min(1.0, (bbox_gt[3] - cby1) / cbh)),
                    )
                    high_labels = compute_binary_labels(nh_high, nw_high, local_gt)
                else:
                    high_labels = torch.zeros(vis_ranges[latest_img_idx][1])

                full_labels = torch.zeros(1, n_total, device=device)
                offset_hi, n_hi = vis_ranges[latest_img_idx]
                full_labels[0, offset_hi:offset_hi + n_hi] = high_labels.to(device)

                attn, loss = self.model.action_head(
                    vis_hidden, anchor, labels=full_labels, mask=full_mask)
                if loss is not None:
                    total_loss = total_loss + loss
                    n_valid += 1

                # Record final prediction
                with torch.no_grad():
                    img_idx, local_idx = identify_attended_image(
                        attn.squeeze(0), vis_ranges)
                    if img_idx < len(grid_dims):
                        nh_a, nw_a = grid_dims[img_idx]
                        lx, ly = token_to_spatial(local_idx, nw_a, nh_a)
                        info = self.builder.image_infos[img_idx]
                        bx1, by1, bx2, by2 = info.global_bbox
                        round_preds.append((
                            bx1 + lx * (bx2 - bx1),
                            by1 + ly * (by2 - by1),
                        ))

                self.metrics["crop_hit"].append(1)
                break  # Target found in crop, stop saccade

            else:
                # --- Saccade: GT not in crop, redirect to unmasked low-res ---
                low_labels = compute_binary_labels(nh0, nw0, bbox_gt)
                low_labels[this_crop_mask] = 0.0  # zero out current crop's patches

                if low_labels.sum() > 0:
                    full_labels = torch.zeros(1, n_total, device=device)
                    full_labels[0, :n_low] = low_labels.to(device)

                    attn, loss = self.model.action_head(
                        vis_hidden, anchor, labels=full_labels, mask=full_mask)
                    if loss is not None:
                        total_loss = total_loss + loss
                        n_valid += 1
                else:
                    # All GT-overlapping patches are masked — skip loss
                    with torch.no_grad():
                        attn, _ = self.model.action_head(
                            vis_hidden, anchor, mask=full_mask)

                self.metrics["crop_hit"].append(0)

                # Get prediction for next round's crop center
                with torch.no_grad():
                    img_idx, local_idx = identify_attended_image(
                        attn.squeeze(0), vis_ranges)
                    if img_idx < len(grid_dims):
                        nh_a, nw_a = grid_dims[img_idx]
                        lx, ly = token_to_spatial(local_idx, nw_a, nh_a)
                        info = self.builder.image_infos[img_idx]
                        bx1, by1, bx2, by2 = info.global_bbox
                        pred_x = bx1 + lx * (bx2 - bx1)
                        pred_y = by1 + ly * (by2 - by1)
                        round_preds.append((pred_x, pred_y))
                    else:
                        break  # Can't determine prediction

        if n_valid > 0:
            total_loss = total_loss / n_valid
        return total_loss, n_valid, round_preds

    # -- train step -----------------------------------------------------------

    def _train_step(self, sample):
        loss, nv, round_preds = self._forward_sample(sample)
        if nv > 0 and loss.requires_grad:
            if self.use_deepspeed:
                self.model_engine.backward(loss)
            else:
                (loss / max(self.args.gradient_accumulation_steps, 1)).backward()

        # Metrics
        gcx = (sample["bbox_gt"][0] + sample["bbox_gt"][2]) / 2
        gcy = (sample["bbox_gt"][1] + sample["bbox_gt"][3]) / 2
        self.metrics["avg_rounds"].append(len(round_preds))
        valid_preds = [p for p in round_preds if p is not None]
        if valid_preds:
            final_px, final_py = valid_preds[-1]
            d = math.sqrt((final_px - gcx)**2 + (final_py - gcy)**2)
            self.metrics["avg_dist"].append(d)
            hit = (sample["bbox_gt"][0] <= final_px <= sample["bbox_gt"][2]
                   and sample["bbox_gt"][1] <= final_py <= sample["bbox_gt"][3])
            self.metrics["hit_rate"].append(1 if hit else 0)
        return loss.item() if nv > 0 else 0.0, nv

    def train_step(self, batch):
        acc_loss = 0.0
        acc_n = 0
        for sample in batch:
            try:
                loss_val, nv = self._train_step(sample)
                if nv > 0:
                    acc_loss += loss_val
                    acc_n += 1
            except Exception:
                traceback.print_exc()
        return acc_loss, acc_n

    # -- main train loop ------------------------------------------------------

    def train(self):
        bs = self.args.per_device_train_batch_size
        ga = max(self.args.gradient_accumulation_steps, 1)
        epochs = int(self.args.num_train_epochs)
        max_steps = getattr(self.args, 'max_steps', -1) or -1

        if self.rank == 0:
            print("=== Saccade Foveation Training (v4: LoRA + ActionHead) ===")
            print(f"  samples={len(self.train_data)}  epochs={epochs}  bs={bs}  ga={ga}")
            print(f"  low_res={self.sa.low_res_max_pixels}  high_res={self.sa.high_res_max_pixels}")
            print(f"  crop_ratio={self.sa.crop_ratio}  max_saccade_rounds={self.sa.max_saccade_rounds}")
            print(f"  action_head_lr={self.sa.action_head_lr}  lora_lr={self.sa.lora_lr}")
            print(f"  lora_r={self.sa.lora_r}  lora_alpha={self.sa.lora_alpha}")
            print(f"  lora_targets={self.sa.lora_target_modules}")
            print(f"  world_size={self.world_size}  deepspeed={self.use_deepspeed}  max_steps={max_steps}")

        if not self.use_deepspeed:
            self.optimizer.zero_grad()
        micro = 0

        for epoch in range(epochs):
            # Deterministic shuffle: same order across all ranks for correct sharding
            epoch_rng = random.Random(42 + epoch)
            epoch_rng.shuffle(self.train_data)
            if self.world_size > 1:
                shard = self.train_data[self.rank::self.world_size]
            else:
                shard = self.train_data
            pbar = tqdm(range(0, len(shard), bs), desc=f"Epoch {epoch+1}/{epochs}",
                        disable=(self.rank != 0))
            for i in pbar:
                loss_val, nv = self.train_step(shard[i:i+bs])
                micro += 1
                if self.use_deepspeed:
                    self.model_engine.step()
                    self.global_step += 1
                elif micro % ga == 0:
                    # Sync gradients across GPUs before optimizer step
                    if self.world_size > 1:
                        for p in self.model.parameters():
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    if any(p.grad is not None for p in self.model.parameters()):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Logging
                if self.rank == 0 and self.global_step % self.args.logging_steps == 0:
                    should_log = self.use_deepspeed or (micro % ga == 0)
                    if should_log:
                        def ar(k):
                            return sum(self.metrics[k][-20:]) / max(len(self.metrics[k][-20:]), 1)
                        lr_val = (self.model_engine.get_lr()[0] if self.use_deepspeed
                                  else self.scheduler.get_last_lr()[0])
                        print(f"  [Step {self.global_step}] loss={loss_val:.4f} "
                              f"hit={ar('hit_rate'):.1%} dist={ar('avg_dist'):.4f} "
                              f"rounds={ar('avg_rounds'):.1f} "
                              f"crop_hit={ar('crop_hit'):.1%} lr={lr_val:.2e}")
                        for key in list(self.metrics.keys()):
                            if len(self.metrics[key]) > 200:
                                self.metrics[key] = self.metrics[key][-200:]

                if self.global_step % self.args.save_steps == 0 and self.global_step > 0:
                    self._save(f"checkpoint-{self.global_step}")

                if max_steps > 0 and self.global_step >= max_steps:
                    if self.rank == 0:
                        print(f"Reached max_steps={max_steps}, stopping.")
                    self._save(f"checkpoint-{self.global_step}")
                    return

        self._save("final")
        if self.rank == 0:
            print(f"Done. Saved to {self.args.output_dir}")

    def _save(self, name):
        if self.world_size > 1:
            dist.barrier()
        if self.rank != 0:
            return
        p = os.path.join(self.args.output_dir, name)
        os.makedirs(p, exist_ok=True)
        self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)
        with open(os.path.join(p, "metrics.json"), "w") as f:
            json.dump({k: v[-100:] for k, v in self.metrics.items()}, f)
        print(f"  Saved: {p}")


# -- Main --------------------------------------------------------------------

def main():
    # Distributed setup (torchrun sets RANK, LOCAL_RANK, WORLD_SIZE)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank() if dist.is_initialized() else 0

    parser = transformers.HfArgumentParser((ScriptArgs, TrainArgs))
    sa, ta = parser.parse_args_into_dataclasses()

    if rank == 0:
        print(f"Building model: {sa.model_name_or_path}")
    model, tokenizer, processor = build_model(
        sa.model_name_or_path,
        lora_r=sa.lora_r,
        lora_alpha=sa.lora_alpha,
        lora_target_modules=sa.lora_target_modules,
        torch_dtype=torch.bfloat16 if ta.bf16 else None,
        gradient_checkpointing=ta.gradient_checkpointing,
    )

    if not (HAS_DEEPSPEED and ta.deepspeed):
        model.to(f"cuda:{local_rank}")

    train_data = load_dataset(sa.data_path, sa.image_folder, sa.max_samples)
    os.makedirs(ta.output_dir, exist_ok=True)

    trainer = SaccadeTrainer(model, tokenizer, train_data, ta, sa)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
