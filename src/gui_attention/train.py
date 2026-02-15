"""
Multi-Precision Foveated Training (SFT + GRPO).

Progressive zoom: each round the model's pointer_pad attention selects a
visual token across ALL images in context.  We identify which image (precision
level) it belongs to, crop around that location, and add a higher-resolution
version.  Terminates when attention lands on a high/ultra-high level image.

Precision levels:
    Level 0 – low    (250 000 px)
    Level 1 – original (1 003 520 px)
    Level 2 – high   (4 000 000 px)
    Level 3 – ultra  (14 000 000 px)

Usage:
    python -m gui_attention.train \
        --model_name_or_path /path/to/GUI-AIMA-3B \
        --data_path /path/to/guiact_bbox.json \
        --image_folder /path/to/images \
        --output_dir /path/to/checkpoints \
        ...
"""

import os
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image, ImageFile
from tqdm import tqdm

import transformers
from transformers import AutoProcessor, AutoTokenizer

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

from transformers import Qwen2_5_VLForConditionalGeneration

from gui_attention.constants import (
    PRECISION_LEVELS, STOP_LEVELS, precision_for_level,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    ADDITIONAL_SPECIAL_TOKENS,
)
from gui_attention.crop import crop_image
from gui_attention.attention import (
    find_image_visual_ranges,
    find_nth_pointer_pad,
    extract_last_layer_attention,
    forward_for_cache,
    identify_attended_image,
    token_to_spatial,
)
from gui_attention.sampling import sample_from_attention
from gui_attention.builder import MultiRoundInputBuilder
from gui_attention.foveation import FoveationLoop


# ── Arguments ─────────────────────────────────────────────────────────────────

@dataclass
class ScriptArgs:
    model_name_or_path: str = field(default="/root/autodl-tmp/models/GUI-AIMA-3B")
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    max_samples: Optional[int] = field(default=None)
    min_pixels: int = field(default=3136)
    max_pixels: int = field(default=PRECISION_LEVELS[0])
    max_rounds: int = field(default=5)
    crop_ratio: float = field(default=0.3)
    initial_level: int = field(default=0, metadata={"help": "Precision level for round 0 (0-3)"})


@dataclass
class TrainArgs(transformers.TrainingArguments):
    training_mode: str = field(default="grpo", metadata={"help": "'grpo' or 'sft'"})
    num_generations: int = field(default=8)
    max_completion_length: int = field(default=256)
    temperature: float = field(default=0.9)
    attn_temperature: float = field(default=1.0)
    beta: float = field(default=0.0)
    epsilon: float = field(default=0.2)
    num_iterations: int = field(default=1)
    optim: str = field(default="adamw_torch")
    gradient_checkpointing: bool = field(default=True)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)


# ── Reward ────────────────────────────────────────────────────────────────────

def position_reward(pred_x, pred_y, bbox_gt, img_w, img_h):
    gt_cx = (bbox_gt[0] + bbox_gt[2]) / 2
    gt_cy = (bbox_gt[1] + bbox_gt[3]) / 2
    dist_px = math.sqrt(((pred_x - gt_cx) * img_w) ** 2 +
                        ((pred_y - gt_cy) * img_h) ** 2)
    bbox_diag = max(math.sqrt(((bbox_gt[2] - bbox_gt[0]) * img_w) ** 2 +
                              ((bbox_gt[3] - bbox_gt[1]) * img_h) ** 2), 1.0)
    return 1.0 / (dist_px / bbox_diag + 1.0)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_dataset(data_path, image_folder, max_samples=None):
    with open(data_path) as f:
        raw = json.load(f)
    if max_samples:
        random.shuffle(raw)
        raw = raw[:max_samples]
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
    print(f"Loaded {len(samples)} samples")
    return samples


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_model_device(model):
    if hasattr(model, 'hf_device_map'):
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, str):
            return torch.device(first_device)
        return torch.device(f"cuda:{first_device}")
    return model.device


def _get_all_visual_indices(input_ids, image_token_id, up_to_pos=None):
    """Get indices of ALL visual tokens, optionally only those before a given position."""
    ranges = find_image_visual_ranges(input_ids, image_token_id)
    indices = []
    range_offsets = []  # (offset_in_combined, n_tokens) per image
    offset = 0
    for vs, ve in ranges:
        if up_to_pos is not None and vs >= up_to_pos:
            break
        indices.append(torch.arange(vs, ve, device=input_ids.device))
        range_offsets.append((offset, ve - vs))
        offset += ve - vs
    if not indices:
        return None, []
    return torch.cat(indices), range_offsets


def _nll_at_target(attn_1d, target_idx, temperature=1.0):
    """Compute negative log probability of target_idx in the attention distribution."""
    if temperature != 1.0:
        logits = torch.log(attn_1d.clamp(min=1e-10)) / temperature
        log_p = F.log_softmax(logits, dim=-1)
    else:
        log_p = torch.log(attn_1d.clamp(min=1e-10))
    return -log_p[target_idx]


def _gt_token_in_image(gt_x, gt_y, image_info, nw, nh):
    """Find the visual token index (within this image) closest to GT.

    Args:
        gt_x, gt_y: GT coordinates in global image space [0,1].
        image_info: ImageInfo with global_bbox.
        nw, nh: grid dimensions of this image.

    Returns:
        flat token index within this image, or None if GT is outside bbox.
    """
    bx1, by1, bx2, by2 = image_info.global_bbox
    bw = bx2 - bx1
    bh = by2 - by1
    if bw <= 0 or bh <= 0:
        return None
    local_x = (gt_x - bx1) / bw
    local_y = (gt_y - by1) / bh
    if not (0 <= local_x <= 1 and 0 <= local_y <= 1):
        return None
    px = max(0, min(round(local_x * nw - 0.5), nw - 1))
    py = max(0, min(round(local_y * nh - 0.5), nh - 1))
    return py * nw + px


# ── Trainer ───────────────────────────────────────────────────────────────────

class MultiRoundTrainer:
    def __init__(self, model, processor, tokenizer, train_data,
                 args: TrainArgs, script_args: ScriptArgs):
        self.processor = processor
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.args = args
        self.sa = script_args
        self.use_deepspeed = HAS_DEEPSPEED and args.deepspeed is not None

        self.builder = MultiRoundInputBuilder(
            script_args.model_name_or_path, tokenizer, script_args.min_pixels,
        )
        self.fov_loop = FoveationLoop(
            max_rounds=script_args.max_rounds,
            crop_ratio=script_args.crop_ratio,
        )

        total_steps = (
            int(args.num_train_epochs) * len(train_data)
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
            self.optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate, weight_decay=args.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7,
            )
            self.rank = 0
            self.world_size = 1

        self.metrics = defaultdict(list)
        self.global_step = 0

    # ── extract attention over all visual tokens for a given pointer_pad ──

    def _extract_attention_for_pointer(self, model, hidden_states, position_ids,
                                        input_ids, round_idx):
        """Extract attention from round_idx's pointer_pad to ALL preceding visual tokens.

        Returns (selected_attn_1d, per_head_attn, best_head, visual_indices,
                 range_offsets, vis_ranges) or None.
        """
        img_tok = model.config.image_token_id
        pp_id = model.config.pointer_pad_token_id
        ptr_pos = find_nth_pointer_pad(input_ids[0], pp_id, round_idx)
        if ptr_pos is None:
            return None

        visual_indices, range_offsets = _get_all_visual_indices(
            input_ids[0], img_tok, up_to_pos=ptr_pos,
        )
        if visual_indices is None:
            return None

        per_head, selected, best_head = extract_last_layer_attention(
            model, hidden_states, position_ids, ptr_pos, visual_indices,
        )

        vis_ranges = find_image_visual_ranges(input_ids[0], img_tok)
        # Only include ranges up to ptr_pos
        vis_ranges = [(vs, ve) for vs, ve in vis_ranges if vs < ptr_pos]

        return selected, per_head, best_head, visual_indices, range_offsets, vis_ranges

    # ── SFT: teacher forcing through precision levels ─────────────────────

    def _sft_forward(self, sample):
        """SFT: teacher forcing with GT coordinates through precision levels.

        Builds rounds L0 → L1 → L2 with crops around GT. Single forward pass.
        For each round's pointer_pad, attention goes over ALL visual tokens
        (from all images in context), and loss targets the GT position in
        the highest-level image at that round.

        Returns (loss, num_rounds, round_preds).
        """
        device = _get_model_device(self.model)
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return torch.tensor(0.0, device=device), 0, []

        bbox_gt = sample["bbox_gt"]
        gt_x = (bbox_gt[0] + bbox_gt[2]) / 2
        gt_y = (bbox_gt[1] + bbox_gt[3]) / 2

        # Build all rounds with teacher forcing
        self.builder.reset()
        level = self.sa.initial_level
        r0_inputs, cur_text, cur_images = self.builder.build_round0(
            sample, sample["instruction"], level=level,
        )
        last_inputs = r0_inputs

        # Add crop rounds until we hit a stop level
        num_sft_rounds = min(self.sa.max_rounds, len(PRECISION_LEVELS) - level)
        for ri in range(1, num_sft_rounds):
            next_level = level + ri
            if next_level >= len(PRECISION_LEVELS):
                break
            cropped, cbbox = crop_image(img, gt_x, gt_y, self.sa.crop_ratio)
            try:
                ri_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                    cur_text, cur_images, cropped, cbbox, level=next_level,
                )
            except Exception:
                break
            last_inputs = ri_inputs
            # Stop building if we've reached a stop level
            if next_level in STOP_LEVELS:
                break

        num_rounds = len(self.builder.image_infos)

        # Single forward pass on the full sequence
        self.model.train()
        inp = {k: v.to(device) for k, v in last_inputs.items()}
        input_ids = inp["input_ids"]
        attention_mask = inp.get("attention_mask", torch.ones_like(input_ids))
        cache = forward_for_cache(
            self.model, input_ids, inp.get("pixel_values"),
            inp.get("image_grid_thw"), attention_mask,
        )

        # Get spatial grid dims per image
        merge = self.model.visual.spatial_merge_size
        grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)

        # Extract attention and compute NLL for each round
        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0
        round_preds = []

        for ri in range(num_rounds):
            result = self._extract_attention_for_pointer(
                self.model, cache["hidden_states"], cache["position_ids"],
                input_ids, round_idx=ri,
            )
            if result is None:
                round_preds.append(None)
                continue

            selected_attn, _, _, visual_indices, range_offsets, vis_ranges = result

            # Target: GT position in the highest-level image at this round (image ri)
            if ri < len(grid_dims):
                nh, nw = grid_dims[ri]
            else:
                round_preds.append(None)
                continue

            target_local = _gt_token_in_image(
                gt_x, gt_y, self.builder.image_infos[ri], nw, nh,
            )
            if target_local is None:
                round_preds.append(None)
                continue

            # Global index in the combined attention vector
            img_offset = range_offsets[ri][0] if ri < len(range_offsets) else 0
            target_global = img_offset + target_local

            nll = _nll_at_target(selected_attn, target_global, temperature=1.0)
            total_loss = total_loss + nll
            n_valid += 1

            # Argmax prediction for metrics
            with torch.no_grad():
                argmax_global = selected_attn.argmax().item()
                # Find which image it's in
                img_idx, local_idx = identify_attended_image(
                    selected_attn, [(0, ro[1]) for ro in range_offsets],
                )
                if img_idx < len(grid_dims):
                    nh_a, nw_a = grid_dims[img_idx]
                    lx, ly = token_to_spatial(local_idx, nw_a, nh_a)
                    info = self.builder.image_infos[img_idx]
                    bx1, by1, bx2, by2 = info.global_bbox
                    ox = bx1 + lx * (bx2 - bx1)
                    oy = by1 + ly * (by2 - by1)
                    round_preds.append((ox, oy))
                else:
                    round_preds.append(None)

        if n_valid > 0:
            total_loss = total_loss / n_valid
        return total_loss, n_valid, round_preds

    # ── GRPO: sampling + policy gradient ──────────────────────────────────

    def _sample_generations(self, sample, num_gen):
        """Sample multi-round foveation trajectories for GRPO."""
        device = _get_model_device(self.model)
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            img_w, img_h = img.size
        except Exception:
            return []

        self.model.eval()
        generations = []

        for _ in range(num_gen):
            self.builder.reset()
            state = self.fov_loop.new_state()
            level = self.sa.initial_level

            # Round 0: full image at initial level
            r0_inputs, cur_text, cur_images = self.builder.build_round0(
                sample, sample["instruction"], level=level,
            )
            last_inputs = r0_inputs

            round_log_probs = []
            round_coords = []  # global coords per round
            round_levels = []  # which level was attended
            round_inputs_list = [r0_inputs]

            for ri in range(self.sa.max_rounds):
                inp = {k: v.to(device) for k, v in last_inputs.items()}
                input_ids = inp["input_ids"]
                attention_mask = inp.get("attention_mask", torch.ones_like(input_ids))

                with torch.no_grad():
                    cache = forward_for_cache(
                        self.model, input_ids, inp.get("pixel_values"),
                        inp.get("image_grid_thw"), attention_mask,
                    )
                    result = self._extract_attention_for_pointer(
                        self.model, cache["hidden_states"], cache["position_ids"],
                        input_ids, round_idx=ri,
                    )
                if result is None:
                    break

                selected_attn, _, best_head, visual_indices, range_offsets, vis_ranges = result

                # Sample from attention
                n_total = selected_attn.numel()
                if self.args.attn_temperature != 1.0:
                    logits = torch.log(selected_attn.clamp(min=1e-10)) / self.args.attn_temperature
                    probs = F.softmax(logits, dim=-1)
                else:
                    probs = selected_attn
                probs = probs.float()
                probs = probs / probs.sum().clamp(min=1e-8)
                dist_cat = torch.distributions.Categorical(probs=probs)
                sampled_idx = dist_cat.sample()
                log_prob = dist_cat.log_prob(sampled_idx)
                round_log_probs.append(log_prob)

                # Identify which image the sampled token is in
                img_idx, local_idx = identify_attended_image(
                    selected_attn,
                    [(0, ro[1]) for ro in range_offsets],
                )
                attended_level = self.builder.image_infos[img_idx].level if img_idx < len(self.builder.image_infos) else 0

                merge = self.model.visual.spatial_merge_size
                grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)
                if img_idx < len(grid_dims):
                    nh, nw = grid_dims[img_idx]
                else:
                    nw = nh = int(math.sqrt(range_offsets[img_idx][1]))

                lx, ly = token_to_spatial(local_idx, nw, nh)
                info = self.builder.image_infos[img_idx]
                bx1, by1, bx2, by2 = info.global_bbox
                ox = bx1 + lx * (bx2 - bx1)
                oy = by1 + ly * (by2 - by1)

                round_coords.append((ox, oy))
                round_levels.append(attended_level)

                # Foveation decision
                decision = self.fov_loop.decide(state, attended_level, ox, oy)

                if decision["action"] == "stop":
                    break

                if decision["action"] == "crop":
                    next_level = decision["level"]
                    cropped, cbbox = crop_image(img, ox, oy, self.sa.crop_ratio)
                    try:
                        ri_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                            cur_text, cur_images, cropped, cbbox, level=next_level,
                        )
                    except Exception:
                        break
                    last_inputs = ri_inputs
                    round_inputs_list.append(ri_inputs)

                if not self.fov_loop.should_continue(state, ri + 1):
                    break

            if not round_coords:
                continue

            final_x, final_y = round_coords[-1]
            reward = position_reward(final_x, final_y, sample["bbox_gt"], img_w, img_h)

            generations.append({
                "reward": reward,
                "round_log_probs": round_log_probs,
                "round_coords": round_coords,
                "round_levels": round_levels,
                "round_inputs_list": round_inputs_list,
                "pred_coord": (final_x, final_y),
                "num_rounds": len(round_coords),
            })

        return generations

    def _compute_loss(self, generations):
        """GRPO loss: advantage-weighted log probability."""
        device = _get_model_device(self.model)
        rewards = torch.tensor([g["reward"] for g in generations], device=device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0
        self.model.train()

        for gen, adv in zip(generations, advantages):
            if gen["pred_coord"] is None or abs(adv.item()) < 1e-8:
                continue

            num_rounds = gen["num_rounds"]
            # Use the last round's inputs (contains all rounds' tokens)
            last_inputs = gen["round_inputs_list"][-1]
            inp = {k: v.to(device) for k, v in last_inputs.items()}
            input_ids = inp["input_ids"]
            attention_mask = inp.get("attention_mask", torch.ones_like(input_ids))

            cache = forward_for_cache(
                self.model, input_ids, inp.get("pixel_values"),
                inp.get("image_grid_thw"), attention_mask,
            )

            diff_lp = torch.tensor(0.0, device=device)
            for ri in range(num_rounds):
                result = self._extract_attention_for_pointer(
                    self.model, cache["hidden_states"], cache["position_ids"],
                    input_ids, round_idx=ri,
                )
                if result is None:
                    continue

                selected_attn, _, _, visual_indices, range_offsets, vis_ranges = result

                # Recompute log probability for the sampled token
                if self.args.attn_temperature != 1.0:
                    logits = torch.log(selected_attn.clamp(min=1e-10)) / self.args.attn_temperature
                    log_p = F.log_softmax(logits, dim=-1)
                else:
                    log_p = torch.log(selected_attn.clamp(min=1e-10))

                # Find which token was sampled at this round
                # (we need to re-identify it from the global coords)
                ox, oy = gen["round_coords"][ri]
                img_idx = min(ri, len(self.builder.image_infos) - 1)
                merge = self.model.visual.spatial_merge_size
                grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)
                if img_idx < len(grid_dims):
                    nh, nw = grid_dims[img_idx]
                else:
                    continue
                info = self.builder.image_infos[img_idx] if img_idx < len(self.builder.image_infos) else None
                if info is None:
                    continue
                bx1, by1, bx2, by2 = info.global_bbox
                bw, bh = bx2 - bx1, by2 - by1
                if bw <= 0 or bh <= 0:
                    continue
                local_x = (ox - bx1) / bw
                local_y = (oy - by1) / bh
                px = max(0, min(round(local_x * nw - 0.5), nw - 1))
                py = max(0, min(round(local_y * nh - 0.5), nh - 1))
                local_idx = py * nw + px

                if img_idx < len(range_offsets):
                    global_idx = range_offsets[img_idx][0] + local_idx
                else:
                    continue

                if global_idx < log_p.numel():
                    diff_lp = diff_lp + log_p[global_idx]

            total_loss = total_loss + (-adv * diff_lp)
            n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid
        return total_loss, n_valid

    # ── train step dispatching ────────────────────────────────────────────

    def _grpo_train_step(self, sample):
        device = _get_model_device(self.model)
        gens = self._sample_generations(sample, self.args.num_generations)
        if not gens:
            return 0.0, 0
        loss, nv = self._compute_loss(gens)
        if nv > 0 and loss.requires_grad:
            if self.use_deepspeed:
                self.model_engine.backward(loss)
            else:
                (loss / max(self.args.gradient_accumulation_steps, 1)).backward()

        # Metrics
        rs = [g["reward"] for g in gens]
        self.metrics["reward"].append(sum(rs) / len(rs))
        self.metrics["reward_std"].append(
            (sum((r - sum(rs)/len(rs))**2 for r in rs) / len(rs)) ** 0.5)
        self.metrics["avg_rounds"].append(
            sum(g["num_rounds"] for g in gens) / len(gens))
        coords = [g["pred_coord"] for g in gens if g["pred_coord"]]
        gcx = (sample["bbox_gt"][0] + sample["bbox_gt"][2]) / 2
        gcy = (sample["bbox_gt"][1] + sample["bbox_gt"][3]) / 2
        if coords:
            dists = [math.sqrt((c[0]-gcx)**2+(c[1]-gcy)**2) for c in coords]
            self.metrics["avg_dist"].append(sum(dists) / len(dists))
            hits = sum(1 for c in coords
                       if sample["bbox_gt"][0]<=c[0]<=sample["bbox_gt"][2]
                       and sample["bbox_gt"][1]<=c[1]<=sample["bbox_gt"][3])
            self.metrics["hit_rate"].append(hits / len(coords))
        levels = [l for g in gens for l in g.get("round_levels", [])]
        if levels:
            self.metrics["avg_level"].append(sum(levels) / len(levels))
        return loss.item() if nv > 0 else 0.0, nv

    def _sft_train_step(self, sample):
        loss, nv, round_preds = self._sft_forward(sample)
        if nv > 0 and loss.requires_grad:
            if self.use_deepspeed:
                self.model_engine.backward(loss)
            else:
                (loss / max(self.args.gradient_accumulation_steps, 1)).backward()

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
        is_sft = self.args.training_mode == "sft"
        for sample in batch:
            try:
                if is_sft:
                    loss_val, nv = self._sft_train_step(sample)
                else:
                    loss_val, nv = self._grpo_train_step(sample)
                if nv > 0:
                    acc_loss += loss_val
                    acc_n += 1
            except Exception as e:
                import traceback; traceback.print_exc()
        return acc_loss, acc_n

    # ── main train loop ──────────────────────────────────────────────────

    def train(self):
        bs = self.args.per_device_train_batch_size
        ga = max(self.args.gradient_accumulation_steps, 1)
        epochs = int(self.args.num_train_epochs)
        max_steps = getattr(self.args, 'max_steps', -1) or -1

        is_sft = self.args.training_mode == "sft"
        if self.rank == 0:
            mode_str = "SFT (teacher forcing)" if is_sft else "GRPO"
            print(f"=== Multi-Precision Foveated {mode_str} ===")
            print(f"  samples={len(self.train_data)}  epochs={epochs}  bs={bs}  ga={ga}")
            print(f"  max_rounds={self.sa.max_rounds}  initial_level={self.sa.initial_level}")
            print(f"  levels={PRECISION_LEVELS}  stop_at={STOP_LEVELS}")
            print(f"  crop_ratio={self.sa.crop_ratio}  lr={self.args.learning_rate}")
            print(f"  deepspeed={self.use_deepspeed}  max_steps={max_steps}")

        if not self.use_deepspeed:
            self.optimizer.zero_grad()
        micro = 0

        for epoch in range(epochs):
            random.shuffle(self.train_data)
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
                        ar = lambda k: sum(self.metrics[k][-20:]) / max(len(self.metrics[k][-20:]), 1)
                        lr_val = (self.model_engine.get_lr()[0] if self.use_deepspeed
                                  else self.scheduler.get_last_lr()[0])
                        if is_sft:
                            print(f"  [Step {self.global_step}] loss={loss_val:.4f} "
                                  f"hit={ar('hit_rate'):.1%} dist={ar('avg_dist'):.4f} "
                                  f"rounds={ar('avg_rounds'):.1f} lr={lr_val:.2e}")
                        else:
                            print(f"  [Step {self.global_step}] loss={loss_val:.4f} "
                                  f"reward={ar('reward'):.3f} (+-{ar('reward_std'):.3f}) "
                                  f"hit={ar('hit_rate'):.1%} dist={ar('avg_dist'):.4f} "
                                  f"rounds={ar('avg_rounds'):.1f} "
                                  f"level={ar('avg_level'):.1f} lr={lr_val:.2e}")
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
        if self.rank != 0:
            return
        p = os.path.join(self.args.output_dir, name)
        os.makedirs(p, exist_ok=True)
        self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)
        self.processor.save_pretrained(p)
        with open(os.path.join(p, "metrics.json"), "w") as f:
            json.dump({k: v[-100:] for k, v in self.metrics.items()}, f)
        print(f"  Saved: {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = transformers.HfArgumentParser((ScriptArgs, TrainArgs))
    sa, ta = parser.parse_args_into_dataclasses()

    print(f"Loading model: {sa.model_name_or_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        sa.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if ta.bf16 else None,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False
    if ta.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad = True

    tokenizer = AutoTokenizer.from_pretrained(sa.model_name_or_path)
    processor = AutoProcessor.from_pretrained(
        sa.model_name_or_path, min_pixels=sa.min_pixels, max_pixels=sa.max_pixels,
    )
    processor.tokenizer = tokenizer

    num_new = tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    if num_new > 0:
        model.resize_token_embeddings(len(tokenizer))
        ie = model.get_input_embeddings().weight.data
        oe = model.get_output_embeddings().weight.data
        ie[-num_new:] = ie[:-num_new].mean(0, keepdim=True)
        oe[-num_new:] = oe[:-num_new].mean(0, keepdim=True)

    model.config.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
    model.config.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
    model.config.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]

    if not (HAS_DEEPSPEED and ta.deepspeed):
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_dataset(sa.data_path, sa.image_folder, sa.max_samples)
    os.makedirs(ta.output_dir, exist_ok=True)

    trainer = MultiRoundTrainer(model, processor, tokenizer, train_data, ta, sa)
    trainer.train()


if __name__ == "__main__":
    main()
