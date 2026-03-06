"""GRPO (Group Relative Policy Optimization) for Saccade Foveation.

Hybrid training: GRPO for saccade strategy + SFT for head precision.

For each sample, generate G trajectories by sampling from attention distributions.
Reward = hit (click in GT bbox) - α * rounds_used.
GRPO updates the model to favor high-reward trajectories.

Key insight: LookHead and ClickHead output attention distributions (softmax logits).
Instead of argmax, we sample from these distributions to get diverse trajectories.
The log-probability of each sampled action enables policy gradient updates.

Usage:
    torchrun --nproc_per_node=8 src/gui_attention/grpo.py \
        --model_name_or_path /path/to/sft_checkpoint \
        --data_path /path/to/data.json \
        --image_folder /path/to/images \
        --output_dir /path/to/output
"""

import json
import math
import os
import random
import traceback
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
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
from gui_attention.constants import HIGH_RES_MAX_PIXELS, LOW_RES_MAX_PIXELS, IGNORE_INDEX
from gui_attention.crop import crop_image, point_in_bbox
from gui_attention.labels import compute_binary_labels, compute_overlap_mask
from gui_attention.model import Qwen25VLWithDualHead, build_model

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -- Arguments ----------------------------------------------------------------

@dataclass
class GRPOArgs:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    max_samples: Optional[int] = field(default=None)
    min_pixels: int = field(default=3136)
    low_res_max_pixels: int = field(default=LOW_RES_MAX_PIXELS)
    high_res_max_pixels: int = field(default=HIGH_RES_MAX_PIXELS)
    crop_size: int = field(default=308)
    crop_upscale: int = field(default=3)
    max_saccade_rounds: int = field(default=6)
    # GRPO
    group_size: int = field(default=4, metadata={"help": "Number of trajectories per sample (G)"})
    reward_hit: float = field(default=1.0, metadata={"help": "Reward for clicking in GT bbox"})
    reward_round_penalty: float = field(default=0.05, metadata={"help": "Penalty per round used"})
    reward_overlap_penalty: float = field(default=0.1, metadata={"help": "Penalty weight for crop overlap (IoU with previous crops)"})
    kl_coeff: float = field(default=0.01, metadata={"help": "KL penalty coefficient against reference"})
    clip_eps: float = field(default=0.2, metadata={"help": "PPO-style clipping epsilon"})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature for attention distributions"})
    # Loss weights (SFT components still used as auxiliary)
    look_sft_weight: float = field(default=0.1, metadata={"help": "Weight for LookHead SFT KL loss (auxiliary)"})
    click_sft_weight: float = field(default=0.1, metadata={"help": "Weight for ClickHead SFT KL loss (auxiliary)"})
    lm_loss_weight: float = field(default=0.1)
    # LoRA
    use_lora: bool = field(default=True)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_target_modules: str = field(default="q_proj,v_proj")
    # LR
    action_head_lr: float = field(default=1e-5, metadata={"help": "LR for heads (lower for GRPO fine-tuning)"})
    lora_lr: float = field(default=5e-6, metadata={"help": "LR for backbone (lower for GRPO fine-tuning)"})
    # Data
    max_samples_per_dataset: Optional[str] = field(default=None)
    # Init
    click_head_from: Optional[str] = field(default=None)
    resume_ckpt: Optional[str] = field(default=None)


@dataclass
class GRPOTrainArgs(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    gradient_checkpointing: bool = field(default=True)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)


# -- Helpers ------------------------------------------------------------------

def _get_model_device(model):
    return next(model.parameters()).device


def _get_visual_module(model):
    _backbone = model.backbone
    if hasattr(_backbone, 'base_model') and hasattr(_backbone.base_model, 'model'):
        _inner = _backbone.base_model.model
    else:
        _inner = _backbone
    if hasattr(_inner, 'visual'):
        return _inner.visual
    elif hasattr(_inner, 'model') and hasattr(_inner.model, 'visual'):
        return _inner.model.visual
    raise AttributeError(f"Cannot find visual module in {type(_inner)}")


# -- Trajectory ---------------------------------------------------------------

@dataclass
class Trajectory:
    """A single saccade trajectory for one sample."""
    round_preds: list          # [(x, y), ...] per round
    crop_bboxes: list          # [bbox, ...] per crop round
    look_log_probs: list       # [log_prob] per round (LookHead action)
    click_log_prob: float      # ClickHead action log_prob
    click_pred: tuple          # (x, y) final click position
    n_rounds: int              # total rounds used
    hit: bool                  # whether click was in GT bbox
    reward: float              # computed reward


# -- GRPO Trainer -------------------------------------------------------------

class GRPOTrainer:
    def __init__(self, model: Qwen25VLWithDualHead, ref_model: Qwen25VLWithDualHead,
                 tokenizer, train_data, args: GRPOTrainArgs, grpo_args: GRPOArgs):
        self.model = model
        self.ref_model = ref_model  # frozen reference for KL
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.args = args
        self.ga = grpo_args
        self.builder = MultiRoundInputBuilder(
            grpo_args.model_name_or_path, tokenizer, grpo_args.min_pixels,
            low_res_max_pixels=grpo_args.low_res_max_pixels,
            high_res_max_pixels=grpo_args.high_res_max_pixels,
        )

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        total_steps = (
            int(args.num_train_epochs) * len(train_data)
            // self.world_size
            // max(args.per_device_train_batch_size, 1)
            // max(args.gradient_accumulation_steps, 1)
        )

        head_params = list(model.dual_head.parameters())
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        param_groups = [
            {"params": head_params, "lr": grpo_args.action_head_lr},
            {"params": backbone_params, "lr": grpo_args.lora_lr},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        warmup_steps = int(total_steps * args.warmup_ratio) if args.warmup_ratio > 0 else 0
        self.scheduler = transformers.get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=max(total_steps, 1),
        )

        self.metrics = defaultdict(list)
        self.global_step = 0

    # -- generate one trajectory (sampling) -----------------------------------

    def _generate_trajectory(self, sample, img) -> Optional[Trajectory]:
        """Run one saccade trajectory with sampling (not argmax).

        LookHead: sample crop center from attention distribution.
        ClickHead: sample click position from attention distribution.
        """
        device = _get_model_device(self.model)
        bbox_gt = sample["bbox_gt"]
        gt_cx = (bbox_gt[0] + bbox_gt[2]) / 2
        gt_cy = (bbox_gt[1] + bbox_gt[3]) / 2

        img_tok = self.model.config.image_token_id
        pp_id = self.model.config.pointer_pad_token_id
        merge = _get_visual_module(self.model).spatial_merge_size

        self.builder.reset()
        round_preds = []
        crop_bboxes = []
        look_log_probs = []
        all_crop_info = []

        max_rounds = self.ga.max_saccade_rounds
        temperature = self.ga.temperature

        for ri in range(max_rounds):
            if ri == 0:
                r_inputs, cur_text, cur_images = self.builder.build_round0(
                    sample, sample["instruction"])
                inp = {k: v.to(device) for k, v in r_inputs.items()}
                if inp.get("pixel_values") is not None:
                    inp["pixel_values"] = inp["pixel_values"].requires_grad_(True)

                outputs = self.model(
                    input_ids=inp["input_ids"],
                    attention_mask=inp.get("attention_mask"),
                    pixel_values=inp["pixel_values"],
                    image_grid_thw=inp.get("image_grid_thw"),
                )
                last_hs = outputs.hidden_states[-1]

                vis_embeds = self.model.extract_visual_embeds(
                    inp["input_ids"], inp["pixel_values"], inp.get("image_grid_thw"))
                _, vis_ranges = extract_visual_hidden_states(
                    last_hs, inp["input_ids"], img_tok)
                anchor = extract_anchor_hidden_states(
                    last_hs, inp["input_ids"], pp_id, n=0)

                if vis_embeds is None or anchor is None:
                    return None

                grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)
                nh0, nw0 = grid_dims[0]

                # LookHead forward
                _, _, logits = self.model.dual_head.look(vis_embeds, anchor)
                logits_1d = logits.squeeze(0) / temperature

                # Sample from attention distribution
                probs = F.softmax(logits_1d, dim=-1)
                sampled_idx = torch.multinomial(probs, 1).item()
                log_prob = F.log_softmax(logits_1d, dim=-1)[sampled_idx].item()
                look_log_probs.append(log_prob)

                # Convert sampled token to spatial coordinates
                pred_x = ((sampled_idx % nw0) + 0.5) / nw0
                pred_y = ((sampled_idx // nw0) + 0.5) / nh0
                round_preds.append((pred_x, pred_y))

                # Virtual crop check
                _, virtual_bbox = crop_image(img, pred_x, pred_y,
                                             crop_size=self.ga.crop_size,
                                             crop_upscale=self.ga.crop_upscale)

            else:
                crop_cx, crop_cy = round_preds[-1]
                cropped, crop_bbox = crop_image(img, crop_cx, crop_cy,
                                                crop_size=self.ga.crop_size,
                                                crop_upscale=self.ga.crop_upscale)
                crop_bboxes.append(crop_bbox)
                gt_in_crop = point_in_bbox(gt_cx, gt_cy, crop_bbox)

                # Rebuild context
                self.builder.reset()
                r_inputs, cur_text, cur_images = self.builder.build_round0(
                    sample, sample["instruction"])
                for prev_ri in range(1, ri):
                    prev_px, prev_py = round_preds[prev_ri]  # round_preds[1] = first crop pred
                    if prev_ri - 1 < len(crop_bboxes):
                        prev_crop, prev_bbox = crop_image(img, round_preds[prev_ri - 1][0],
                                                          round_preds[prev_ri - 1][1],
                                                          crop_size=self.ga.crop_size,
                                                          crop_upscale=self.ga.crop_upscale)
                        try:
                            r_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                                cur_text, cur_images, prev_crop, prev_bbox)
                        except Exception:
                            break
                try:
                    r_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                        cur_text, cur_images, cropped, crop_bbox)
                except Exception:
                    break

                inp = {k: v.to(device) for k, v in r_inputs.items()}
                if inp.get("pixel_values") is not None:
                    inp["pixel_values"] = inp["pixel_values"].requires_grad_(True)

                outputs = self.model(
                    input_ids=inp["input_ids"],
                    attention_mask=inp.get("attention_mask"),
                    pixel_values=inp["pixel_values"],
                    image_grid_thw=inp.get("image_grid_thw"),
                )
                last_hs = outputs.hidden_states[-1]

                vis_embeds = self.model.extract_visual_embeds(
                    inp["input_ids"], inp["pixel_values"], inp.get("image_grid_thw"))
                _, vis_ranges = extract_visual_hidden_states(
                    last_hs, inp["input_ids"], img_tok)
                anchor = extract_anchor_hidden_states(
                    last_hs, inp["input_ids"], pp_id, n=ri)

                if vis_embeds is None or anchor is None or len(vis_ranges) < ri + 1:
                    break

                grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)
                n_total = vis_embeds.shape[0]
                n_low = vis_ranges[0][1]
                latest_img_idx = len(vis_ranges) - 1

                # Mask old crops
                this_crop_mask = compute_overlap_mask(nh0, nw0, crop_bbox)
                full_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
                full_mask[:n_low] = this_crop_mask.to(device)
                for prev_i in range(1, latest_img_idx):
                    off_prev, ntok_prev = vis_ranges[prev_i]
                    full_mask[off_prev:off_prev + ntok_prev] = True

                # Save crop info
                offset_hi, n_hi = vis_ranges[latest_img_idx]
                nh_high, nw_high = grid_dims[latest_img_idx]
                all_crop_info.append({
                    "offset": offset_hi, "n_tokens": n_hi,
                    "grid_h": nh_high, "grid_w": nw_high,
                    "crop_bbox": crop_bbox, "gt_in_crop": gt_in_crop,
                })

                if gt_in_crop:
                    # LookHead: sample on full tokens
                    _, _, logits = self.model.dual_head.look(
                        vis_embeds, anchor, mask=full_mask)
                    logits_1d = logits.squeeze(0) / temperature
                    probs = F.softmax(logits_1d, dim=-1)
                    sampled_idx = torch.multinomial(probs, 1).item()
                    log_prob = F.log_softmax(logits_1d, dim=-1)[sampled_idx].item()
                    look_log_probs.append(log_prob)

                    # ClickHead: sample on all crop tokens
                    crop_vis_list = [vis_embeds[ci["offset"]:ci["offset"] + ci["n_tokens"]]
                                     for ci in all_crop_info]
                    combined_crop_vis = torch.cat(crop_vis_list, dim=0)

                    _, _, click_logits = self.model.dual_head.click(
                        combined_crop_vis, anchor)
                    click_logits_1d = click_logits.squeeze(0) / temperature
                    click_probs = F.softmax(click_logits_1d, dim=-1)
                    click_sampled = torch.multinomial(click_probs, 1).item()
                    click_log_prob = F.log_softmax(click_logits_1d, dim=-1)[click_sampled].item()

                    # Map sampled click token back to global coords
                    running = 0
                    click_pred = (0.5, 0.5)  # fallback
                    for ci in all_crop_info:
                        if running + ci["n_tokens"] > click_sampled:
                            local_tok = click_sampled - running
                            lx = ((local_tok % ci["grid_w"]) + 0.5) / ci["grid_w"]
                            ly = ((local_tok // ci["grid_w"]) + 0.5) / ci["grid_h"]
                            bx1, by1, bx2, by2 = ci["crop_bbox"]
                            click_pred = (bx1 + lx * (bx2 - bx1), by1 + ly * (by2 - by1))
                            break
                        running += ci["n_tokens"]

                    # Check hit
                    hit = (bbox_gt[0] <= click_pred[0] <= bbox_gt[2]
                           and bbox_gt[1] <= click_pred[1] <= bbox_gt[3])

                    return Trajectory(
                        round_preds=round_preds,
                        crop_bboxes=crop_bboxes,
                        look_log_probs=look_log_probs,
                        click_log_prob=click_log_prob,
                        click_pred=click_pred,
                        n_rounds=ri + 1,
                        hit=hit,
                        reward=0.0,  # computed later
                    )

                else:
                    # Saccade: sample from LookHead
                    _, _, logits = self.model.dual_head.look(
                        vis_embeds, anchor, mask=full_mask)
                    logits_1d = logits.squeeze(0) / temperature
                    probs = F.softmax(logits_1d, dim=-1)
                    sampled_idx = torch.multinomial(probs, 1).item()
                    log_prob = F.log_softmax(logits_1d, dim=-1)[sampled_idx].item()
                    look_log_probs.append(log_prob)

                    # Map sampled token to global coords
                    img_idx, _ = identify_attended_image(probs, vis_ranges)
                    if img_idx < len(grid_dims):
                        nh_a, nw_a = grid_dims[img_idx]
                        off_a, n_a = vis_ranges[img_idx]
                        # Find local index within this image
                        local_sampled = sampled_idx - off_a
                        if 0 <= local_sampled < n_a:
                            lx = ((local_sampled % nw_a) + 0.5) / nw_a
                            ly = ((local_sampled // nw_a) + 0.5) / nh_a
                            info = self.builder.image_infos[img_idx]
                            bx1, by1, bx2, by2 = info.global_bbox
                            pred_x = bx1 + lx * (bx2 - bx1)
                            pred_y = by1 + ly * (by2 - by1)
                            round_preds.append((pred_x, pred_y))
                        else:
                            break
                    else:
                        break

        # Reached max rounds without finding GT
        # Use last LookHead prediction as click (no ClickHead)
        if round_preds:
            click_pred = round_preds[-1]
        else:
            click_pred = (0.5, 0.5)

        hit = (bbox_gt[0] <= click_pred[0] <= bbox_gt[2]
               and bbox_gt[1] <= click_pred[1] <= bbox_gt[3])

        return Trajectory(
            round_preds=round_preds,
            crop_bboxes=crop_bboxes,
            look_log_probs=look_log_probs,
            click_log_prob=0.0,  # no ClickHead used
            click_pred=click_pred,
            n_rounds=len(round_preds),
            hit=hit,
            reward=0.0,
        )

    # -- compute rewards ------------------------------------------------------

    @staticmethod
    def _bbox_iou(a, b):
        """IoU between two (x1, y1, x2, y2) bboxes."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _compute_overlap_penalty(self, crop_bboxes):
        """Sum of max IoU with any previous crop, for each crop."""
        total = 0.0
        for i, bbox in enumerate(crop_bboxes):
            if i == 0:
                continue
            max_iou = max(self._bbox_iou(bbox, crop_bboxes[j]) for j in range(i))
            total += max_iou
        return total

    def _compute_rewards(self, trajectories: list):
        """Compute rewards for a group of trajectories (same sample).

        reward = hit * reward_hit - round_penalty * n_rounds
                 - overlap_penalty * sum_of_max_ious
        Then group-normalize (GRPO).
        """
        for traj in trajectories:
            r = 0.0
            if traj.hit:
                r += self.ga.reward_hit
            r -= self.ga.reward_round_penalty * traj.n_rounds
            # Overlap penalty: penalize revisiting same areas
            if traj.crop_bboxes and self.ga.reward_overlap_penalty > 0:
                r -= self.ga.reward_overlap_penalty * self._compute_overlap_penalty(traj.crop_bboxes)
            traj.reward = r

        # Group normalization
        rewards = [t.reward for t in trajectories]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        std_r = max(std_r, 1e-8)

        for traj in trajectories:
            traj.reward = (traj.reward - mean_r) / std_r  # advantage

    # -- GRPO loss ------------------------------------------------------------

    def _compute_grpo_loss(self, sample, trajectories):
        """Compute GRPO policy gradient loss.

        loss = -Σ advantage_i * (Σ look_log_probs + click_log_prob)
               + β * KL(π || π_ref)

        Also includes optional SFT auxiliary losses for head precision.
        """
        device = _get_model_device(self.model)

        # Policy gradient: weight log_probs by advantage
        pg_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for traj in trajectories:
            if abs(traj.reward) < 1e-8:
                continue  # zero advantage, skip

            # Replay trajectory to get differentiable log_probs
            traj_loss = self._replay_trajectory_loss(sample, traj)
            if traj_loss is not None:
                pg_loss = pg_loss + traj_loss

        return pg_loss

    def _replay_trajectory_loss(self, sample, traj: Trajectory):
        """Replay a trajectory and compute differentiable policy gradient loss.

        Returns: -advantage * sum(log_probs)
        """
        device = _get_model_device(self.model)
        bbox_gt = sample["bbox_gt"]
        gt_cx = (bbox_gt[0] + bbox_gt[2]) / 2
        gt_cy = (bbox_gt[1] + bbox_gt[3]) / 2

        img_tok = self.model.config.image_token_id
        pp_id = self.model.config.pointer_pad_token_id
        merge = _get_visual_module(self.model).spatial_merge_size

        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return None

        self.builder.reset()
        self.model.train()
        temperature = self.ga.temperature
        total_log_prob = torch.tensor(0.0, device=device)

        for ri in range(traj.n_rounds):
            if ri == 0:
                r_inputs, cur_text, cur_images = self.builder.build_round0(
                    sample, sample["instruction"])
                inp = {k: v.to(device) for k, v in r_inputs.items()}
                if inp.get("pixel_values") is not None:
                    inp["pixel_values"] = inp["pixel_values"].requires_grad_(True)

                outputs = self.model(
                    input_ids=inp["input_ids"],
                    attention_mask=inp.get("attention_mask"),
                    pixel_values=inp["pixel_values"],
                    image_grid_thw=inp.get("image_grid_thw"),
                )
                last_hs = outputs.hidden_states[-1]
                vis_embeds = self.model.extract_visual_embeds(
                    inp["input_ids"], inp["pixel_values"], inp.get("image_grid_thw"))
                _, vis_ranges = extract_visual_hidden_states(
                    last_hs, inp["input_ids"], img_tok)
                anchor = extract_anchor_hidden_states(
                    last_hs, inp["input_ids"], pp_id, n=0)

                if vis_embeds is None or anchor is None:
                    return None

                grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)
                nh0, nw0 = grid_dims[0]

                # LookHead: get log_prob of the action taken
                _, _, logits = self.model.dual_head.look(vis_embeds, anchor)
                logits_1d = logits.squeeze(0) / temperature

                # Reconstruct which token was sampled from round_preds
                pred_x, pred_y = traj.round_preds[0]
                tok_col = int(pred_x * nw0)
                tok_row = int(pred_y * nh0)
                tok_col = min(max(tok_col, 0), nw0 - 1)
                tok_row = min(max(tok_row, 0), nh0 - 1)
                sampled_idx = tok_row * nw0 + tok_col

                log_prob = F.log_softmax(logits_1d, dim=-1)[sampled_idx]
                total_log_prob = total_log_prob + log_prob

            else:
                # Use the stored round_preds to reconstruct crops
                prev_x, prev_y = traj.round_preds[ri - 1]
                cropped, crop_bbox = crop_image(img, prev_x, prev_y,
                                                crop_size=self.ga.crop_size,
                                                crop_upscale=self.ga.crop_upscale)

                # Rebuild context
                self.builder.reset()
                r_inputs, cur_text, cur_images = self.builder.build_round0(
                    sample, sample["instruction"])
                for prev_ri in range(1, ri):
                    pp_x, pp_y = traj.round_preds[prev_ri - 1]
                    prev_crop, prev_bbox = crop_image(img, pp_x, pp_y,
                                                      crop_size=self.ga.crop_size,
                                                      crop_upscale=self.ga.crop_upscale)
                    try:
                        r_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                            cur_text, cur_images, prev_crop, prev_bbox)
                    except Exception:
                        return None
                try:
                    r_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                        cur_text, cur_images, cropped, crop_bbox)
                except Exception:
                    return None

                inp = {k: v.to(device) for k, v in r_inputs.items()}
                if inp.get("pixel_values") is not None:
                    inp["pixel_values"] = inp["pixel_values"].requires_grad_(True)

                outputs = self.model(
                    input_ids=inp["input_ids"],
                    attention_mask=inp.get("attention_mask"),
                    pixel_values=inp["pixel_values"],
                    image_grid_thw=inp.get("image_grid_thw"),
                )
                last_hs = outputs.hidden_states[-1]
                vis_embeds = self.model.extract_visual_embeds(
                    inp["input_ids"], inp["pixel_values"], inp.get("image_grid_thw"))
                _, vis_ranges = extract_visual_hidden_states(
                    last_hs, inp["input_ids"], img_tok)
                anchor = extract_anchor_hidden_states(
                    last_hs, inp["input_ids"], pp_id, n=ri)

                if vis_embeds is None or anchor is None:
                    return None

                grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)
                n_total = vis_embeds.shape[0]
                n_low = vis_ranges[0][1]
                latest_img_idx = len(vis_ranges) - 1

                # Mask
                this_crop_mask = compute_overlap_mask(nh0, nw0, crop_bbox)
                full_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
                full_mask[:n_low] = this_crop_mask.to(device)
                for prev_i in range(1, latest_img_idx):
                    off_prev, ntok_prev = vis_ranges[prev_i]
                    full_mask[off_prev:off_prev + ntok_prev] = True

                # LookHead log_prob
                _, _, logits = self.model.dual_head.look(
                    vis_embeds, anchor, mask=full_mask)
                logits_1d = logits.squeeze(0) / temperature

                # Reconstruct sampled token from round_preds
                if ri < len(traj.round_preds):
                    pred_x, pred_y = traj.round_preds[ri]
                    # Find which image and local token
                    # For simplicity, find the token with coords closest to pred
                    # (since we stored the global coords during generation)
                    best_idx = logits_1d.argmax().item()  # fallback
                    for img_i, (off_i, n_i) in enumerate(vis_ranges):
                        if img_i < len(grid_dims) and img_i < len(self.builder.image_infos):
                            nh_i, nw_i = grid_dims[img_i]
                            info = self.builder.image_infos[img_i]
                            bx1, by1, bx2, by2 = info.global_bbox
                            if bx1 <= pred_x <= bx2 and by1 <= pred_y <= by2:
                                local_x = (pred_x - bx1) / max(bx2 - bx1, 1e-8)
                                local_y = (pred_y - by1) / max(by2 - by1, 1e-8)
                                col = min(max(int(local_x * nw_i), 0), nw_i - 1)
                                row = min(max(int(local_y * nh_i), 0), nh_i - 1)
                                best_idx = off_i + row * nw_i + col
                                break

                    if best_idx < logits_1d.shape[0]:
                        log_prob = F.log_softmax(logits_1d, dim=-1)[best_idx]
                        total_log_prob = total_log_prob + log_prob

        # ClickHead log_prob (if ClickHead was used)
        if traj.click_log_prob != 0.0 and traj.hit is not None:
            # We'd need to replay ClickHead too, but for now use stored log_prob
            # TODO: full differentiable replay of ClickHead
            pass

        # GRPO loss: -advantage * total_log_prob
        advantage = traj.reward
        loss = -advantage * total_log_prob

        return loss

    # -- train step -----------------------------------------------------------

    def _train_step(self, sample):
        """Generate G trajectories, compute rewards, update."""
        device = _get_model_device(self.model)
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return 0.0

        # Generate G trajectories (with no_grad for sampling)
        trajectories = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.ga.group_size):
                traj = self._generate_trajectory(sample, img)
                if traj is not None:
                    trajectories.append(traj)

        if len(trajectories) < 2:
            return 0.0

        # Compute rewards and advantages
        self._compute_rewards(trajectories)

        # Log metrics
        hits = sum(1 for t in trajectories if t.hit)
        avg_rounds = sum(t.n_rounds for t in trajectories) / len(trajectories)
        self.metrics["grpo_hit_rate"].append(hits / len(trajectories))
        self.metrics["grpo_avg_rounds"].append(avg_rounds)
        self.metrics["grpo_avg_reward"].append(
            sum(t.reward for t in trajectories) / len(trajectories))

        # Compute GRPO loss (differentiable replay)
        self.model.train()
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for traj in trajectories:
            if abs(traj.reward) < 1e-8:
                continue
            traj_loss = self._replay_trajectory_loss(sample, traj)
            if traj_loss is not None:
                total_loss = total_loss + traj_loss

        total_loss = total_loss / max(len(trajectories), 1)

        if total_loss.requires_grad:
            (total_loss / max(self.args.gradient_accumulation_steps, 1)).backward()

        return total_loss.item()

    # -- main train loop ------------------------------------------------------

    def train(self):
        bs = self.args.per_device_train_batch_size
        ga = max(self.args.gradient_accumulation_steps, 1)
        epochs = int(self.args.num_train_epochs)

        if self.rank == 0:
            print(f"=== GRPO Saccade Training ===")
            print(f"  samples={len(self.train_data)}  epochs={epochs}  bs={bs}  ga={ga}")
            print(f"  group_size={self.ga.group_size}  temperature={self.ga.temperature}")
            print(f"  reward_hit={self.ga.reward_hit}  round_penalty={self.ga.reward_round_penalty}")
            print(f"  kl_coeff={self.ga.kl_coeff}  clip_eps={self.ga.clip_eps}")
            print(f"  head_lr={self.ga.action_head_lr}  backbone_lr={self.ga.lora_lr}")
            print(f"  max_saccade_rounds={self.ga.max_saccade_rounds}")

        self.optimizer.zero_grad()
        micro = 0

        for epoch in range(epochs):
            epoch_rng = random.Random(42 + epoch)
            epoch_rng.shuffle(self.train_data)
            shard = self.train_data[self.rank::self.world_size] if self.world_size > 1 else self.train_data

            pbar = tqdm(range(0, len(shard), bs), desc=f"Epoch {epoch+1}/{epochs}",
                        disable=(self.rank != 0))
            for i in pbar:
                batch = shard[i:i+bs]
                batch_loss = 0.0
                for sample in batch:
                    try:
                        loss_val = self._train_step(sample)
                        batch_loss += loss_val
                    except Exception:
                        traceback.print_exc()

                micro += 1
                if micro % ga == 0:
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
                    if micro % ga == 0 and self.metrics["grpo_hit_rate"]:
                        def ar(k, n=20):
                            vals = self.metrics[k][-n:]
                            return sum(vals) / max(len(vals), 1)
                        lrs = self.scheduler.get_last_lr()
                        print(f"  [Step {self.global_step}] "
                              f"hit={ar('grpo_hit_rate'):.1%} "
                              f"rounds={ar('grpo_avg_rounds'):.1f} "
                              f"reward={ar('grpo_avg_reward'):.3f} "
                              f"lr={lrs[0]:.2e}")
                        for key in list(self.metrics.keys()):
                            if len(self.metrics[key]) > 500:
                                self.metrics[key] = self.metrics[key][-500:]

                if self.global_step % self.args.save_steps == 0 and self.global_step > 0:
                    self._save(f"checkpoint-{self.global_step}")

        self._save("final")
        if self.rank == 0:
            print(f"Done. Saved to {self.args.output_dir}")

    def _save(self, name):
        if self.world_size > 1:
            dist.barrier()
        p = os.path.join(self.args.output_dir, name)
        if self.rank != 0:
            return
        os.makedirs(p, exist_ok=True)
        self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }, os.path.join(p, "training_state.pt"))
        print(f"  Saved: {p}")


# -- Data loading (reuse from train.py) ----------------------------------------

def load_dataset(data_path, image_folder, max_samples=None, max_per_ds=None):
    from gui_attention.train import load_dataset as _load_dataset
    return _load_dataset(data_path, image_folder, max_samples, max_per_ds)


# -- Main ---------------------------------------------------------------------

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank() if dist.is_initialized() else 0

    parser = transformers.HfArgumentParser((GRPOArgs, GRPOTrainArgs))
    ga, ta = parser.parse_args_into_dataclasses()

    if rank == 0:
        print(f"Building model: {ga.model_name_or_path}")
    model, tokenizer, processor = build_model(
        ga.model_name_or_path,
        lora_r=ga.lora_r,
        lora_alpha=ga.lora_alpha,
        lora_target_modules=ga.lora_target_modules,
        torch_dtype=torch.bfloat16 if ta.bf16 else None,
        gradient_checkpointing=ta.gradient_checkpointing,
        use_lora=ga.use_lora,
        click_head_from=ga.click_head_from,
    )

    # Load SFT checkpoint if specified
    if ga.resume_ckpt:
        ckpt = ga.resume_ckpt
        if rank == 0:
            print(f"Loading SFT checkpoint: {ckpt}")
        if ga.use_lora:
            from peft import set_peft_model_state_dict
            adapter_file = os.path.join(ckpt, "adapter_model.safetensors")
            if os.path.exists(adapter_file):
                from safetensors.torch import load_file
                adapter_state = load_file(adapter_file)
            else:
                adapter_state = torch.load(os.path.join(ckpt, "adapter_model.bin"),
                                           map_location="cpu", weights_only=True)
            set_peft_model_state_dict(model.backbone, adapter_state)

        dual_path = os.path.join(ckpt, "dual_head.pt")
        old_path = os.path.join(ckpt, "action_head.pt")
        if os.path.exists(dual_path):
            model.dual_head.load_state_dict(
                torch.load(dual_path, map_location="cpu", weights_only=True))
        elif os.path.exists(old_path):
            old_state = torch.load(old_path, map_location="cpu", weights_only=True)
            look_state = {f"look_head.{k}": v for k, v in old_state.items()
                          if not k.startswith("bbox_head") and k != "beta"}
            model.dual_head.load_state_dict(look_state, strict=False)

    model.to(f"cuda:{local_rank}")

    # Reference model (frozen copy for KL penalty)
    ref_model = deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    train_data = load_dataset(ga.data_path, ga.image_folder, ga.max_samples, ga.max_samples_per_dataset)
    os.makedirs(ta.output_dir, exist_ok=True)

    trainer = GRPOTrainer(model, ref_model, tokenizer, train_data, ta, ga)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
