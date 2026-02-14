"""
Multi-Round Progressive GRPO Training with Attention Sampling.

Coarse-to-fine foveation: each round crops around the previous prediction
at increasing resolution, appending the crop as a new temporal image in the
Qwen2.5-VL mrope framework.  Log-probs accumulate across rounds; reward is
computed from the final-round prediction only.

Precision levels (two-stage):
    Round 1  – low   (max_pixels ≈ 1 003 520)
    Round 2+ – high  (max_pixels ≈ 5 760 000)

Token layout per round (appended to the running sequence):
    <crop_image_tokens> [Zoomed …] <|im_start|>assistant<|recipient|>os
    pyautogui.click(<pointer_start><pointer_pad><pointer_end>)

The causal mask guarantees that each round's pointer_pad only attends to
tokens up to (and including) itself, so a single forward pass on the full
sequence gives the same per-round attention as incremental passes.

Usage:
    python train_grpo_multi_round.py \
        --model_name_or_path /root/autodl-tmp/models/GUI-AIMA-3B \
        --data_path /root/autodl-tmp/data/GUI-Actor/guiact_bbox.json \
        --image_folder /root/autodl-tmp/data/GUI-Actor/images/GUIAct/web_imgs \
        --output_dir /root/autodl-tmp/checkpoints/grpo_multi_round \
        ...
"""

import os
import sys
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
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

# Path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../Experiments/GUI-AIMA/src"))

from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.constants import (
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    ADDITIONAL_SPECIAL_TOKENS,
)

from gui_attention.constants import PRECISION_LOW, precision_for_round
from gui_attention.crop import crop_image, get_patch_bbox, point_in_bbox
from gui_attention.attention import (
    get_round_attention, forward_for_attention, extract_attention_for_round,
)
from gui_attention.sampling import sample_from_attention, compute_round_log_prob
from gui_attention.builder import MultiRoundInputBuilder


# ── Arguments ─────────────────────────────────────────────────────────────────
@dataclass
class ScriptArgs:
    model_name_or_path: str = field(default="/root/autodl-tmp/models/GUI-AIMA-3B")
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    max_samples: Optional[int] = field(default=None)
    min_pixels: int = field(default=3136)
    # Round-1 max_pixels; later rounds use precision_for_round()
    max_pixels: int = field(default=PRECISION_LOW)
    max_rounds: int = field(default=5)
    crop_ratio: float = field(default=0.3)


@dataclass
class TrainArgs(transformers.TrainingArguments):
    training_mode: str = field(default="grpo", metadata={"help": "Training mode: 'grpo' or 'sft'"})
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



# ── Trainer ───────────────────────────────────────────────────────────────────

def _get_model_device(model):
    """Get the input device for a model (works with device_map='auto')."""
    if hasattr(model, 'hf_device_map'):
        # device_map model: find the device of the first module
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, str):
            return torch.device(first_device)
        return torch.device(f"cuda:{first_device}")
    return model.device


class MultiRoundGRPOTrainer:
    def __init__(self, model, processor, tokenizer, train_data, args: TrainArgs, script_args: ScriptArgs):
        self.processor = processor
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.args = args
        self.sa = script_args
        self.use_deepspeed = HAS_DEEPSPEED and args.deepspeed is not None

        self.builder = MultiRoundInputBuilder(script_args.model_name_or_path, tokenizer, script_args.min_pixels)

        total_steps = (
            int(args.num_train_epochs) * len(train_data)
            // max(args.per_device_train_batch_size, 1)
            // max(args.gradient_accumulation_steps, 1)
        )
        if args.max_steps and args.max_steps > 0:
            total_steps = min(total_steps, args.max_steps)

        if self.use_deepspeed:
            # DeepSpeed handles optimizer, scheduler, gradient accumulation
            self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=model,
                model_parameters=[p for p in model.parameters() if p.requires_grad],
                config=args.deepspeed,
            )
            self.model = self.model_engine.module
            self.rank = dist.get_rank() if dist.is_initialized() else 0
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
            print(f"  [Rank {self.rank}] DeepSpeed initialized, world_size={self.world_size}")
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

    # ── sampling phase (no grad) ──────────────────────────────────────────

    def _sample_generations(self, sample, num_gen):
        """For each generation, run multi-round foveation and collect log_probs."""
        device = _get_model_device(self.model)
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
            img_w, img_h = img.size
        except Exception:
            return []

        # Round 0 inputs (shared across generations)
        r0_inputs, r0_text, r0_images = self.builder.build_round0(sample, precision_for_round(0))
        r0_dev = {k: v.to(device) for k, v in r0_inputs.items()}

        self.model.eval()
        with torch.no_grad():
            attn0 = get_round_attention(
                self.model, r0_dev["input_ids"], r0_dev.get("pixel_values"),
                r0_dev.get("image_grid_thw"),
                r0_dev.get("attention_mask", torch.ones_like(r0_dev["input_ids"])),
                round_idx=0,
            )
        if attn0 is None:
            return []

        generations = []
        for _ in range(num_gen):
            round_log_probs = []
            round_coords = []        # original-image normalised coords
            round_local_coords = []  # coords local to that round's image
            round_inputs_list = []
            round_crop_bboxes = [None]  # round 0 has no crop bbox

            # Round 0 sample
            px, py, lp, _ = sample_from_attention(
                attn0["attn_weights"], attn0["n_width"], attn0["n_height"],
                temperature=self.args.attn_temperature,
            )
            round_log_probs.append(lp)
            round_coords.append((px, py))
            round_local_coords.append((px, py))
            round_inputs_list.append(r0_inputs)

            prev_x, prev_y = px, py
            prev_nw, prev_nh = attn0["n_width"], attn0["n_height"]
            cur_text, cur_images = r0_text, list(r0_images)

            for ri in range(1, self.sa.max_rounds):
                cropped, cbbox = crop_image(img, prev_x, prev_y, self.sa.crop_ratio)
                try:
                    ri_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                        cur_text, cur_images, cropped, cbbox, ri,
                    )
                except Exception:
                    break

                ri_dev = {k: v.to(device) for k, v in ri_inputs.items()}
                with torch.no_grad():
                    attn_ri = get_round_attention(
                        self.model, ri_dev["input_ids"], ri_dev.get("pixel_values"),
                        ri_dev.get("image_grid_thw"),
                        ri_dev.get("attention_mask", torch.ones_like(ri_dev["input_ids"])),
                        round_idx=ri,
                    )
                if attn_ri is None:
                    break

                cx, cy, lp_r, _ = sample_from_attention(
                    attn_ri["attn_weights"], attn_ri["n_width"], attn_ri["n_height"],
                    temperature=self.args.attn_temperature,
                )
                # map back to original coords
                bx1, by1, bx2, by2 = cbbox
                ox = bx1 + cx * (bx2 - bx1)
                oy = by1 + cy * (by2 - by1)

                round_log_probs.append(lp_r)
                round_coords.append((ox, oy))
                round_local_coords.append((cx, cy))
                round_inputs_list.append(ri_inputs)
                round_crop_bboxes.append(cbbox)

                # Convergence: does the new prediction fall within the
                # previous round's predicted patch (in original-image coords)?
                # The previous round's patch bbox in its own local coords:
                prev_patch_local = get_patch_bbox(
                    round_local_coords[-2][0], round_local_coords[-2][1],
                    prev_nw, prev_nh,
                )
                # Map previous patch bbox to original image coords
                if round_crop_bboxes[-2] is not None:
                    # Previous round was a crop
                    pb = round_crop_bboxes[-2]
                    pw = pb[2] - pb[0]
                    ph = pb[3] - pb[1]
                    prev_patch_orig = (
                        pb[0] + prev_patch_local[0] * pw,
                        pb[1] + prev_patch_local[1] * ph,
                        pb[0] + prev_patch_local[2] * pw,
                        pb[1] + prev_patch_local[3] * ph,
                    )
                else:
                    # Previous round was the full image (round 0)
                    prev_patch_orig = prev_patch_local

                # Only allow convergence from round 2+ (round 0=low, round 1+=high,
                # so ri>=2 means we've seen high-res at least once before)
                if ri >= 2 and point_in_bbox(ox, oy, prev_patch_orig):
                    break

                prev_x, prev_y = ox, oy
                prev_nw, prev_nh = attn_ri["n_width"], attn_ri["n_height"]

            final_x, final_y = round_coords[-1]
            reward = position_reward(
                final_x, final_y, sample["bbox_gt"], img_w, img_h,
            )

            generations.append({
                "reward": reward,
                "total_log_prob": sum(round_log_probs),
                "round_log_probs": round_log_probs,
                "round_coords": round_coords,
                "round_local_coords": round_local_coords,
                "round_inputs_list": round_inputs_list,
                "round_crop_bboxes": round_crop_bboxes,
                "pred_coord": (final_x, final_y),
                "num_rounds": len(round_coords),
            })

        return generations

    # ── loss phase (with grad) ────────────────────────────────────────────

    def _compute_loss(self, generations):
        device = _get_model_device(self.model)
        rewards = torch.tensor([g["reward"] for g in generations], device=device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0
        self.model.train()
        temp = self.args.attn_temperature

        # Round 0: one forward pass shared across all generations
        # (all generations use the same round-0 inputs)
        r0_inputs = generations[0]["round_inputs_list"][0]
        inp0 = {k: v.to(device) for k, v in r0_inputs.items()}
        input_ids_r0 = inp0["input_ids"]
        attention_mask_r0 = inp0.get("attention_mask", torch.ones_like(input_ids_r0))
        r0_cache = forward_for_attention(
            self.model, input_ids_r0, inp0.get("pixel_values"),
            inp0.get("image_grid_thw"), attention_mask_r0,
        )
        r0_attn = extract_attention_for_round(
            self.model, input_ids_r0, inp0.get("image_grid_thw"),
            attention_mask_r0, r0_cache, round_idx=0,
        )

        for gen, adv in zip(generations, advantages):
            if gen["pred_coord"] is None or abs(adv.item()) < 1e-8 or gen["num_rounds"] == 0:
                continue

            diff_lp = torch.tensor(0.0, device=device)
            num_rounds = gen["num_rounds"]

            # Round 0: reuse shared attention
            if r0_attn is not None:
                diff_lp = diff_lp + compute_round_log_prob(
                    r0_attn["attn_weights"], gen["round_local_coords"][0],
                    r0_attn["n_width"], r0_attn["n_height"], temp,
                )

            # Rounds 1+: single forward pass using last round's inputs.
            # The causal mask guarantees each round's pointer_pad attention
            # is identical to processing only up to that round's tokens,
            # so we can extract all rounds from one forward pass.
            if num_rounds > 1:
                last_inputs = gen["round_inputs_list"][-1]
                inp = {k: v.to(device) for k, v in last_inputs.items()}
                input_ids = inp["input_ids"]
                attention_mask = inp.get("attention_mask", torch.ones_like(input_ids))
                cache = forward_for_attention(
                    self.model, input_ids, inp.get("pixel_values"),
                    inp.get("image_grid_thw"), attention_mask,
                )
                for ri in range(1, num_rounds):
                    attn = extract_attention_for_round(
                        self.model, input_ids, inp.get("image_grid_thw"),
                        attention_mask, cache, ri,
                    )
                    if attn is None:
                        continue
                    diff_lp = diff_lp + compute_round_log_prob(
                        attn["attn_weights"], gen["round_local_coords"][ri],
                        attn["n_width"], attn["n_height"], temp,
                    )

            total_loss = total_loss + (-adv * diff_lp)
            n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid
        return total_loss, n_valid

    # ── SFT: supervised fine-tuning with GT attention ────────────────────

    def _sft_forward(self, sample):
        """SFT forward: teacher-forcing with GT coordinates.

        Crops around GT each round, builds the full multi-round sequence,
        then does a single forward pass and computes NLL loss for all rounds.
        Returns (loss, num_rounds, per_round_predictions).
        """
        device = _get_model_device(self.model)
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return torch.tensor(0.0, device=device), 0, []

        bbox_gt = sample["bbox_gt"]
        gt_x = (bbox_gt[0] + bbox_gt[2]) / 2
        gt_y = (bbox_gt[1] + bbox_gt[3]) / 2

        # Build all rounds upfront (teacher forcing: crop around GT)
        r0_inputs, cur_text, cur_images = self.builder.build_round0(
            sample, precision_for_round(0),
        )
        round_local_gts = [(gt_x, gt_y)]  # round 0: GT in full image coords
        round_crop_bboxes = [None]
        last_inputs = r0_inputs

        for ri in range(1, self.sa.max_rounds):
            cropped, cbbox = crop_image(img, gt_x, gt_y, self.sa.crop_ratio)
            try:
                ri_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                    cur_text, cur_images, cropped, cbbox, ri,
                )
            except Exception:
                break
            bx1, by1, bx2, by2 = cbbox
            local_gt_x = (gt_x - bx1) / (bx2 - bx1)
            local_gt_y = (gt_y - by1) / (by2 - by1)
            round_local_gts.append((local_gt_x, local_gt_y))
            round_crop_bboxes.append(cbbox)
            last_inputs = ri_inputs

        num_rounds = len(round_local_gts)

        # Single forward pass on the last round's inputs (causal mask
        # ensures each round's attention is identical to incremental passes)
        self.model.train()
        inp = {k: v.to(device) for k, v in last_inputs.items()}
        input_ids = inp["input_ids"]
        attention_mask = inp.get("attention_mask", torch.ones_like(input_ids))
        cache = forward_for_attention(
            self.model, input_ids, inp.get("pixel_values"),
            inp.get("image_grid_thw"), attention_mask,
        )

        # Extract attention and compute NLL for each round
        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0
        round_preds = []  # (pred_x_orig, pred_y_orig) per round for metrics
        for ri in range(num_rounds):
            attn = extract_attention_for_round(
                self.model, input_ids, inp.get("image_grid_thw"),
                attention_mask, cache, round_idx=ri,
            )
            if attn is None:
                round_preds.append(None)
                continue
            lp = compute_round_log_prob(
                attn["attn_weights"], round_local_gts[ri],
                attn["n_width"], attn["n_height"], temperature=1.0,
            )
            total_loss = total_loss - lp  # NLL
            n_valid += 1

            # Argmax prediction for metrics (no grad needed, detach)
            with torch.no_grad():
                p = attn["attn_weights"].squeeze(0).float()
                idx = p.argmax().item()
                nw, nh = attn["n_width"], attn["n_height"]
                local_px = (idx % nw + 0.5) / nw
                local_py = (idx // nw + 0.5) / nh
                if round_crop_bboxes[ri] is not None:
                    bx1, by1, bx2, by2 = round_crop_bboxes[ri]
                    orig_px = bx1 + local_px * (bx2 - bx1)
                    orig_py = by1 + local_py * (by2 - by1)
                else:
                    orig_px, orig_py = local_px, local_py
                round_preds.append((orig_px, orig_py))

        if n_valid > 0:
            total_loss = total_loss / n_valid
        return total_loss, n_valid, round_preds

    # ── train step / loop ─────────────────────────────────────────────────

    def _grpo_train_step(self, sample):
        """GRPO: sample generations → compute advantage-weighted loss."""
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

        # GRPO metrics
        rs = [g["reward"] for g in gens]
        self.metrics["reward"].append(sum(rs) / len(rs))
        self.metrics["reward_max"].append(max(rs))
        self.metrics["reward_min"].append(min(rs))
        self.metrics["reward_std"].append(
            (sum((r - sum(rs)/len(rs))**2 for r in rs) / len(rs)) ** 0.5)
        self.metrics["avg_rounds"].append(
            sum(g["num_rounds"] for g in gens) / len(gens))
        self.metrics["max_rounds"].append(
            max(g["num_rounds"] for g in gens))
        coords = [g["pred_coord"] for g in gens if g["pred_coord"]]
        gcx = (sample["bbox_gt"][0] + sample["bbox_gt"][2]) / 2
        gcy = (sample["bbox_gt"][1] + sample["bbox_gt"][3]) / 2
        if coords:
            dists = [math.sqrt((c[0]-gcx)**2+(c[1]-gcy)**2) for c in coords]
            self.metrics["avg_dist"].append(sum(dists) / len(dists))
            self.metrics["min_dist"].append(min(dists))
            hits = [1 for c in coords
                    if sample["bbox_gt"][0]<=c[0]<=sample["bbox_gt"][2]
                    and sample["bbox_gt"][1]<=c[1]<=sample["bbox_gt"][3]]
            self.metrics["hit_rate"].append(len(hits) / len(coords))
            converged = [g for g in gens if g["num_rounds"] < self.sa.max_rounds]
            self.metrics["convergence_rate"].append(len(converged) / len(gens))
            self.metrics["advantage_spread"].append(max(rs) - min(rs))
            for g in gens:
                for ri, (rx, ry) in enumerate(g["round_coords"]):
                    key = f"round_{ri}_hit"
                    if key not in self.metrics:
                        self.metrics[key] = []
                    self.metrics[key].append(
                        1 if (sample["bbox_gt"][0]<=rx<=sample["bbox_gt"][2]
                              and sample["bbox_gt"][1]<=ry<=sample["bbox_gt"][3]) else 0)
        for g in gens:
            rc = g["round_coords"]
            if len(rc) > 0:
                for ri, (rx, ry) in enumerate(rc):
                    d = math.sqrt((rx - gcx)**2 + (ry - gcy)**2)
                    key = f"round_{ri}_dist"
                    if key not in self.metrics:
                        self.metrics[key] = []
                    self.metrics[key].append(d)
                if self.global_step % 50 == 0:
                    round_dists = [math.sqrt((rx-gcx)**2+(ry-gcy)**2) for rx, ry in rc]
                    prog = " -> ".join(f"R{i}:{d:.4f}" for i, d in enumerate(round_dists))
                    hit_str = "HIT" if (sample["bbox_gt"][0]<=rc[-1][0]<=sample["bbox_gt"][2]
                                        and sample["bbox_gt"][1]<=rc[-1][1]<=sample["bbox_gt"][3]) else "MISS"
                    print(f"  [Round Progress] step={self.global_step} {prog} {hit_str}")
        return loss.item() if nv > 0 else 0.0, nv

    def _sft_train_step(self, sample):
        """SFT: teacher-forcing with GT coordinates."""
        loss, nv, round_preds = self._sft_forward(sample)
        if nv > 0 and loss.requires_grad:
            if self.use_deepspeed:
                self.model_engine.backward(loss)
            else:
                (loss / max(self.args.gradient_accumulation_steps, 1)).backward()

        # SFT metrics
        gcx = (sample["bbox_gt"][0] + sample["bbox_gt"][2]) / 2
        gcy = (sample["bbox_gt"][1] + sample["bbox_gt"][3]) / 2
        self.metrics["avg_rounds"].append(len(round_preds))
        valid_preds = [p for p in round_preds if p is not None]
        if valid_preds:
            final_px, final_py = valid_preds[-1]
            dist = math.sqrt((final_px - gcx)**2 + (final_py - gcy)**2)
            self.metrics["avg_dist"].append(dist)
            hit = (sample["bbox_gt"][0] <= final_px <= sample["bbox_gt"][2]
                   and sample["bbox_gt"][1] <= final_py <= sample["bbox_gt"][3])
            self.metrics["hit_rate"].append(1 if hit else 0)
            for ri, pred in enumerate(round_preds):
                if pred is None:
                    continue
                d = math.sqrt((pred[0] - gcx)**2 + (pred[1] - gcy)**2)
                key_d = f"round_{ri}_dist"
                key_h = f"round_{ri}_hit"
                if key_d not in self.metrics:
                    self.metrics[key_d] = []
                if key_h not in self.metrics:
                    self.metrics[key_h] = []
                self.metrics[key_d].append(d)
                self.metrics[key_h].append(
                    1 if (sample["bbox_gt"][0] <= pred[0] <= sample["bbox_gt"][2]
                          and sample["bbox_gt"][1] <= pred[1] <= sample["bbox_gt"][3]) else 0)
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

    def train(self):
        bs = self.args.per_device_train_batch_size
        ga = max(self.args.gradient_accumulation_steps, 1)
        epochs = int(self.args.num_train_epochs)
        max_steps = getattr(self.args, 'max_steps', -1) or -1

        is_sft = self.args.training_mode == "sft"
        if self.rank == 0:
            mode_str = "SFT (teacher forcing)" if is_sft else "GRPO"
            print(f"=== Multi-Round Progressive {mode_str} ===")
            print(f"  samples={len(self.train_data)}  epochs={epochs}  bs={bs}  ga={ga}")
            if not is_sft:
                print(f"  generations={self.args.num_generations}  max_rounds={self.sa.max_rounds}")
            else:
                print(f"  max_rounds={self.sa.max_rounds}")
            print(f"  crop_ratio={self.sa.crop_ratio}")
            print(f"  attn_temp={self.args.attn_temperature}  lr={self.args.learning_rate}")
            print(f"  deepspeed={self.use_deepspeed}  world_size={self.world_size}  max_steps={max_steps}")

        if not self.use_deepspeed:
            self.optimizer.zero_grad()
        micro = 0

        for epoch in range(epochs):
            random.shuffle(self.train_data)
            # Shard data across ranks for distributed training
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
                    # DeepSpeed handles gradient accumulation internally
                    self.model_engine.step()
                    self.global_step += 1
                elif micro % ga == 0:
                    if any(p.grad is not None for p in self.model.parameters()):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Logging, saving, and stopping (shared by both DeepSpeed and non-DeepSpeed)
                if self.rank == 0 and self.global_step % self.args.logging_steps == 0:
                    should_log = self.use_deepspeed or (micro % ga == 0)
                    if should_log:
                        ar = lambda k: sum(self.metrics[k][-20:]) / max(len(self.metrics[k][-20:]),1)
                        lr_val = (self.model_engine.get_lr()[0] if self.use_deepspeed
                                  else self.scheduler.get_last_lr()[0])
                        if is_sft:
                            print(f"  [Step {self.global_step}] loss={loss_val:.4f} "
                                  f"hit={ar('hit_rate'):.1%} dist={ar('avg_dist'):.4f} "
                                  f"rounds={ar('avg_rounds'):.1f} "
                                  f"lr={lr_val:.2e}")
                        else:
                            print(f"  [Step {self.global_step}] loss={loss_val:.4f} "
                                  f"reward={ar('reward'):.3f} (+-{ar('reward_std'):.3f}) "
                                  f"hit={ar('hit_rate'):.1%} dist={ar('avg_dist'):.4f} "
                                  f"rounds={ar('avg_rounds'):.1f}/{ar('max_rounds'):.0f} "
                                  f"conv={ar('convergence_rate'):.0%} "
                                  f"lr={lr_val:.2e}")
                        # Per-round distance + hit summary
                        rd_parts = []
                        for ri in range(self.sa.max_rounds):
                            key_d = f"round_{ri}_dist"
                            key_h = f"round_{ri}_hit"
                            if key_d in self.metrics and self.metrics[key_d]:
                                d_str = f"d={ar(key_d):.4f}"
                                h_str = f"h={ar(key_h):.0%}" if key_h in self.metrics and self.metrics[key_h] else ""
                                rd_parts.append(f"R{ri}({d_str},{h_str})")
                        if rd_parts:
                            print(f"           rounds: {' -> '.join(rd_parts)}")
                        if not is_sft:
                            print(f"           spread={ar('advantage_spread'):.3f} "
                                  f"min_dist={ar('min_dist'):.4f}")
                        # Trim metrics to prevent unbounded memory growth
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
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
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
    for attr in ('query_topk', 'kl_query_weighting', 'part_query_weighting', 'layer_wise_query_weighting'):
        if not hasattr(model.config, attr):
            setattr(model.config, attr, 1 if attr == 'query_topk' else False)

    # For DeepSpeed, don't move to device — deepspeed.initialize() handles it
    if not (HAS_DEEPSPEED and ta.deepspeed):
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_dataset(sa.data_path, sa.image_folder, sa.max_samples)
    os.makedirs(ta.output_dir, exist_ok=True)

    trainer = MultiRoundGRPOTrainer(model, processor, tokenizer, train_data, ta, sa)
    trainer.train()


if __name__ == "__main__":
    main()
