"""
Multi-Round Progressive GRPO Training with Attention Sampling.

Coarse-to-fine foveation: each round crops around the previous prediction
at increasing resolution, appending the crop as a new temporal image in the
Qwen2.5-VL mrope framework.  Log-probs accumulate across rounds; reward is
computed from the final-round prediction only.

Precision levels:
    Round 1  – low   (max_pixels ≈ 1 003 520)
    Round 2  – mid   (max_pixels ≈ 3 000 000)
    Round 3+ – high  (max_pixels ≈ 5 760 000)

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
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm

import transformers
from transformers import AutoProcessor, AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True

# GUI-AIMA imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../Experiments/GUI-AIMA/src"))

from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.constants import (
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    ADDITIONAL_SPECIAL_TOKENS,
    chat_template,
    grounding_system_message,
)
from gui_aima.inference import calculate_attention_from_qk
from qwen_vl_utils import process_vision_info

# ── Precision levels ──────────────────────────────────────────────────────────
PRECISION_LOW  = 1_003_520   # ~56×56 × 320 patches
PRECISION_MID  = 3_000_000
PRECISION_HIGH = 5_760_000   # ~56×56 × ~1836 patches

def precision_for_round(round_idx: int) -> int:
    """Return max_pixels for a given 0-based round index."""
    if round_idx == 0:
        return PRECISION_LOW
    elif round_idx == 1:
        return PRECISION_MID
    else:
        return PRECISION_HIGH


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
    format_reward_value: float = field(default=0.05)
    max_rounds: int = field(default=5)
    convergence_threshold: float = field(default=0.02)
    crop_ratio: float = field(default=0.3)


@dataclass
class TrainArgs(transformers.TrainingArguments):
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


# ── Crop ──────────────────────────────────────────────────────────────────────

def crop_image(image: Image.Image, cx_norm, cy_norm, crop_ratio):
    """Crop around normalised centre. Returns (cropped_pil, (x1,y1,x2,y2) normalised)."""
    W, H = image.size
    cw, ch = int(W * crop_ratio), int(H * crop_ratio)
    cx, cy = int(cx_norm * W), int(cy_norm * H)
    x1 = max(0, cx - cw // 2)
    y1 = max(0, cy - ch // 2)
    x2 = min(W, x1 + cw)
    y2 = min(H, y1 + ch)
    if x2 - x1 < cw:
        x1 = max(0, x2 - cw)
    if y2 - y1 < ch:
        y1 = max(0, y2 - ch)
    return image.crop((x1, y1, x2, y2)), (x1 / W, y1 / H, x2 / W, y2 / H)


# ── Attention helpers ─────────────────────────────────────────────────────────

def _find_nth_image_visual_range(input_ids, image_token_id, n):
    """Return (start, end) indices of the n-th (0-based) contiguous block of image tokens."""
    ids = input_ids.tolist()
    blocks = []
    in_block = False
    start = 0
    for i, tid in enumerate(ids):
        if tid == image_token_id:
            if not in_block:
                start = i
                in_block = True
        else:
            if in_block:
                blocks.append((start, i))
                in_block = False
    if in_block:
        blocks.append((start, len(ids)))
    if n < len(blocks):
        return blocks[n]
    return None


def _find_nth_pointer_pad(input_ids, pointer_pad_id, n):
    """Return index of the n-th pointer_pad token (0-based)."""
    if isinstance(pointer_pad_id, list):
        pad_set = set(pointer_pad_id)
    else:
        pad_set = {pointer_pad_id}
    count = 0
    for i, tid in enumerate(input_ids.tolist()):
        if tid in pad_set:
            if count == n:
                return i
            count += 1
    return None


def get_round_attention(
    model, input_ids, pixel_values, image_grid_thw, attention_mask,
    round_idx: int,
):
    """
    Forward-pass the *full* multi-round sequence and extract the attention
    distribution for a specific round's pointer_pad over that round's visual
    tokens.

    Returns dict with attn_weights (1, n_vis), n_width, n_height  –  or None.
    """
    device = input_ids.device
    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id
    ps_id = model.config.pointer_start_token_id

    # Find the round_idx-th image block
    vis_range = _find_nth_image_visual_range(input_ids[0], img_tok, round_idx)
    if vis_range is None:
        return None
    vis_start, vis_end = vis_range
    visual_indices = torch.arange(vis_start, vis_end, device=device)

    # Find the round_idx-th pointer_pad
    target_pos = _find_nth_pointer_pad(input_ids[0], pp_id, round_idx)
    if target_pos is None:
        return None
    target_indices = torch.tensor([target_pos], device=device)

    # Query indices: text tokens between last visual token of this image block
    # and the pointer_start that precedes this pointer_pad
    # Find the pointer_start just before target_pos
    ps_positions = (input_ids[0] == ps_id).nonzero(as_tuple=False).squeeze(-1)
    ps_before = ps_positions[ps_positions < target_pos]
    query_end = ps_before[-1].item() if len(ps_before) > 0 else target_pos
    query_start = vis_end  # first text token after the image block

    query_indices = torch.arange(query_start, query_end, device=device)
    if model.config.part_query_weighting and len(query_indices) > 12:
        query_indices = query_indices[:-12]
    if query_indices.numel() == 0:
        query_indices = torch.arange(max(0, target_pos - 10), target_pos, device=device)

    merged_indices = torch.cat([query_indices, target_indices], dim=0)

    # Forward
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=attention_mask,
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        output_hidden_states=True,
    )

    hs_per_layer = list(outputs.hidden_states)
    calculated_attention = calculate_attention_from_qk(
        model=model,
        all_hidden_states=[hs_per_layer],
        all_position_ids=position_ids,
        query_indices=merged_indices,
        all_attention_mask=attention_mask,
    )

    # Query importance weighting
    all_layer_hs = torch.stack(hs_per_layer[1:], dim=0)
    sample_hs = all_layer_hs[:, 0, :, :]
    q_hs = F.normalize(sample_hs[:, query_indices, :], dim=-1)
    v_hs = F.normalize(sample_hs[:, visual_indices, :], dim=-1)
    sim = torch.einsum('lqd,lvd->lqv', q_hs, v_hs)
    attn_per_query = sim.sum(dim=-1)

    topk_query_indices = None
    global_pattern = None
    if not getattr(model.config, 'kl_query_weighting', False):
        k = getattr(model.config, 'query_topk', 1)
        agg = attn_per_query.sum(dim=0)
        _, topk_query_indices = torch.topk(agg, min(k, len(query_indices)), largest=True)
    else:
        global_pattern = attn_per_query.sum(dim=0).softmax(dim=-1)

    attn_weights, _ = model.multi_patch_pointer_head_attention(
        query_indices, visual_indices, target_indices,
        calculated_attention[0],
        topk_query_indices, global_pattern,
        batch_idx=0,
    )

    # Spatial dims of this round's image
    merge = model.visual.spatial_merge_size
    # image_grid_thw may have multiple rows (one per image).  Use round_idx-th row.
    if round_idx < image_grid_thw.shape[0]:
        _, nh, nw = (image_grid_thw[round_idx] // merge).tolist()
    else:
        # fallback: infer from visual_indices count
        n_vis = visual_indices.numel()
        nw = nh = int(math.sqrt(n_vis))

    return {"attn_weights": attn_weights, "n_width": int(nw), "n_height": int(nh)}


def sample_from_attention(attn_weights, n_w, n_h, temperature=1.0):
    if temperature != 1.0:
        logits = torch.log(attn_weights.clamp(min=1e-10)) / temperature
        probs = F.softmax(logits, dim=-1)
    else:
        probs = attn_weights
    dist = torch.distributions.Categorical(probs=probs.squeeze(0))
    idx = dist.sample()
    lp = dist.log_prob(idx)
    px = idx.item() % n_w
    py = idx.item() // n_w
    return (px + 0.5) / n_w, (py + 0.5) / n_h, lp, idx.item()


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


# ── Multi-round input builder ─────────────────────────────────────────────────

PLACEHOLDER_SUFFIX = (
    "<|im_start|>assistant<|recipient|>os\n"
    "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
)


class MultiRoundInputBuilder:
    """Incrementally builds the multi-round conversation and tokenises it.

    Round 0 (full image):
        system + user[image, instruction] + placeholder

    Round k (crop):
        ... previous ... + "\n" + <crop_image> [Zoomed …] + placeholder
    """

    def __init__(self, model_path: str, tokenizer, min_pixels: int):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.min_pixels = min_pixels
        self._processor_cache: Dict[int, AutoProcessor] = {}

    def _get_processor(self, max_pixels: int):
        if max_pixels not in self._processor_cache:
            p = AutoProcessor.from_pretrained(
                self.model_path,
                min_pixels=self.min_pixels,
                max_pixels=max_pixels,
            )
            p.tokenizer = self.tokenizer
            self._processor_cache[max_pixels] = p
        return self._processor_cache[max_pixels]

    def build_round0(self, sample: dict, max_pixels: int):
        """Build round-0 inputs (full image)."""
        conv = [
            {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": sample["image_path"]},
                {"type": "text", "text": sample["instruction"]},
            ]},
        ]
        text = self._get_processor(max_pixels).apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False, chat_template=chat_template,
        )
        text += PLACEHOLDER_SUFFIX
        images, _ = process_vision_info(conv)
        inputs = self._get_processor(max_pixels)(text=[text], images=images, return_tensors="pt", padding=True)
        return inputs, text, images

    def extend_with_crop(self, prev_text: str, prev_images: list,
                         crop_pil: Image.Image, crop_bbox: tuple,
                         round_idx: int):
        """Append a crop round to the existing conversation text & images."""
        max_px = precision_for_round(round_idx)
        # Append a newline + image placeholder + zoom annotation + new placeholder
        # We insert the <image> tag which process_vision_info will match to the next image.
        zoom_text = (
            f"\n<|im_start|>user\n<image>"
            f"[Zoomed region round {round_idx+1} around "
            f"({crop_bbox[0]:.2f},{crop_bbox[1]:.2f})-({crop_bbox[2]:.2f},{crop_bbox[3]:.2f})]"
            f"<|im_end|>\n"
            + PLACEHOLDER_SUFFIX
        )
        new_text = prev_text + zoom_text
        new_images = prev_images + [crop_pil]

        proc = self._get_processor(max_px)
        inputs = proc(text=[new_text], images=new_images, return_tensors="pt", padding=True)
        return inputs, new_text, new_images


# ── Trainer ───────────────────────────────────────────────────────────────────

class MultiRoundGRPOTrainer:
    def __init__(self, model, processor, tokenizer, train_data, args: TrainArgs, script_args: ScriptArgs):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.args = args
        self.sa = script_args

        self.builder = MultiRoundInputBuilder(script_args.model_name_or_path, tokenizer, script_args.min_pixels)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate, weight_decay=args.weight_decay,
        )
        total_steps = (
            int(args.num_train_epochs) * len(train_data)
            // max(args.per_device_train_batch_size, 1)
            // max(args.gradient_accumulation_steps, 1)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7,
        )
        self.metrics = defaultdict(list)
        self.global_step = 0

    # ── sampling phase (no grad) ──────────────────────────────────────────

    def _sample_generations(self, sample, num_gen):
        """For each generation, run multi-round foveation and collect log_probs."""
        device = self.model.device
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

                d = math.sqrt((ox - prev_x) ** 2 + (oy - prev_y) ** 2)
                prev_x, prev_y = ox, oy
                if d < self.sa.convergence_threshold:
                    break

            final_x, final_y = round_coords[-1]
            reward = self.sa.format_reward_value + position_reward(
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
        device = self.model.device
        rewards = torch.tensor([g["reward"] for g in generations], device=device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0
        self.model.train()

        for gen, adv in zip(generations, advantages):
            if gen["pred_coord"] is None or abs(adv.item()) < 1e-8 or gen["num_rounds"] == 0:
                continue

            diff_lp = torch.tensor(0.0, device=device)

            for ri, (ri_inputs, local_c) in enumerate(
                zip(gen["round_inputs_list"], gen["round_local_coords"])
            ):
                inp = {k: v.to(device) for k, v in ri_inputs.items()}
                attn = get_round_attention(
                    self.model, inp["input_ids"], inp.get("pixel_values"),
                    inp.get("image_grid_thw"),
                    inp.get("attention_mask", torch.ones_like(inp["input_ids"])),
                    round_idx=ri,
                )
                if attn is None:
                    continue

                nw, nh = attn["n_width"], attn["n_height"]
                lx, ly = local_c
                px = max(0, min(round(lx * nw - 0.5), nw - 1))
                py = max(0, min(round(ly * nh - 0.5), nh - 1))
                sidx = py * nw + px

                aw = attn["attn_weights"]
                if self.args.attn_temperature != 1.0:
                    logits = torch.log(aw.clamp(min=1e-10)) / self.args.attn_temperature
                    log_p = F.log_softmax(logits, dim=-1)
                else:
                    log_p = torch.log(aw.clamp(min=1e-10))

                diff_lp = diff_lp + log_p[0, sidx]

            total_loss = total_loss + (-adv * diff_lp)
            n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid
        return total_loss, n_valid

    # ── train step / loop ─────────────────────────────────────────────────

    def train_step(self, batch):
        acc_loss = 0.0
        acc_n = 0
        for sample in batch:
            try:
                gens = self._sample_generations(sample, self.args.num_generations)
                if not gens:
                    continue
                loss, nv = self._compute_loss(gens)
                if nv > 0 and loss.requires_grad:
                    (loss / max(self.args.gradient_accumulation_steps, 1)).backward()
                    acc_loss += loss.item()
                    acc_n += 1
                # metrics
                rs = [g["reward"] for g in gens]
                self.metrics["reward"].append(sum(rs) / len(rs))
                self.metrics["avg_rounds"].append(
                    sum(g["num_rounds"] for g in gens) / len(gens))
                coords = [g["pred_coord"] for g in gens if g["pred_coord"]]
                if coords:
                    gcx = (sample["bbox_gt"][0] + sample["bbox_gt"][2]) / 2
                    gcy = (sample["bbox_gt"][1] + sample["bbox_gt"][3]) / 2
                    self.metrics["avg_dist"].append(
                        sum(math.sqrt((c[0]-gcx)**2+(c[1]-gcy)**2) for c in coords) / len(coords))
                    self.metrics["hit_rate"].append(
                        sum(1 for c in coords
                            if sample["bbox_gt"][0]<=c[0]<=sample["bbox_gt"][2]
                            and sample["bbox_gt"][1]<=c[1]<=sample["bbox_gt"][3]) / len(coords))
            except Exception as e:
                import traceback; traceback.print_exc()
        return acc_loss, acc_n

    def train(self):
        bs = self.args.per_device_train_batch_size
        ga = max(self.args.gradient_accumulation_steps, 1)
        epochs = int(self.args.num_train_epochs)

        print("=== Multi-Round Progressive GRPO ===")
        print(f"  samples={len(self.train_data)}  epochs={epochs}  bs={bs}  ga={ga}")
        print(f"  generations={self.args.num_generations}  max_rounds={self.sa.max_rounds}")
        print(f"  crop_ratio={self.sa.crop_ratio}  convergence={self.sa.convergence_threshold}")
        print(f"  attn_temp={self.args.attn_temperature}  lr={self.args.learning_rate}")

        self.optimizer.zero_grad()
        micro = 0

        for epoch in range(epochs):
            random.shuffle(self.train_data)
            pbar = tqdm(range(0, len(self.train_data), bs), desc=f"Epoch {epoch+1}/{epochs}")
            for i in pbar:
                loss_val, nv = self.train_step(self.train_data[i:i+bs])
                micro += 1
                if micro % ga == 0:
                    if any(p.grad is not None for p in self.model.parameters()):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.args.logging_steps == 0:
                        ar = lambda k: sum(self.metrics[k][-20:]) / max(len(self.metrics[k][-20:]),1)
                        print(f"  [Step {self.global_step}] loss={loss_val:.4f} "
                              f"reward={ar('reward'):.3f} dist={ar('avg_dist'):.4f} "
                              f"hit={ar('hit_rate'):.2%} rounds={ar('avg_rounds'):.1f} "
                              f"lr={self.scheduler.get_last_lr()[0]:.2e}")

                    if self.global_step % self.args.save_steps == 0:
                        self._save(f"checkpoint-{self.global_step}")

        self._save("final")
        print(f"Done. Saved to {self.args.output_dir}")

    def _save(self, name):
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

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_dataset(sa.data_path, sa.image_folder, sa.max_samples)
    os.makedirs(ta.output_dir, exist_ok=True)

    trainer = MultiRoundGRPOTrainer(model, processor, tokenizer, train_data, ta, sa)
    trainer.train()


if __name__ == "__main__":
    main()
