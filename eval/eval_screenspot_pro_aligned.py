"""
ScreenSpot-Pro evaluation script ALIGNED with multi-round GRPO training.

Key alignment with train_grpo_multi_round.py:
  1. Uses attention-based prediction (argmax instead of sampling)
  2. Uses MultiRoundInputBuilder for identical input construction
  3. Uses same convergence criterion (point_in_bbox, ri >= 2)
  4. Uses same precision levels (low → high)
  5. Uses get_round_attention() for attention extraction

Usage:
  # Multi-round attention-based eval (aligned with training)
  python eval/eval_screenspot_pro_aligned.py \
      --model_name_or_path /path/to/checkpoint \
      --rounds 5 --crop_ratio 0.3

  # Single-round eval
  python eval/eval_screenspot_pro_aligned.py \
      --model_name_or_path /path/to/checkpoint \
      --rounds 1 --resolution high
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Add project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.inference import inference, calculate_attention_from_qk
from gui_aima.utils import do_boxes_overlap
from gui_aima.constants import (
    ADDITIONAL_SPECIAL_TOKENS,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    chat_template,
    grounding_system_message,
)
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Precision levels (same as training)
# ---------------------------------------------------------------------------
PRECISION_LOW = 1_003_520
PRECISION_MID = 3_000_000
PRECISION_HIGH = 5_760_000

PLACEHOLDER_SUFFIX = (
    "<|im_start|>assistant<|recipient|>os\n"
    "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
)


# ---------------------------------------------------------------------------
# GUI-AIMA original system prompt
# ---------------------------------------------------------------------------
GROUNDING_SYSTEM_MESSAGE = (
    "You are a GUI agent. Given a screenshot of the current GUI and a human "
    "instruction, your task is to locate the screen element that corresponds "
    "to the instruction. You should output a PyAutoGUI action that performs a "
    "click on the correct position. To indicate the click location, we will "
    "use some special tokens, which is used to refer to a visual patch later. "
    "For example, you can output: pyautogui.click(<your_special_token_here>)."
)


# ---------------------------------------------------------------------------
# Helpers (from training code)
# ---------------------------------------------------------------------------

def crop_image(image, cx_norm, cy_norm, crop_ratio):
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


def get_patch_bbox(px_norm, py_norm, n_width, n_height):
    """Return the normalised bbox of the patch containing (px_norm, py_norm)."""
    col = min(int(px_norm * n_width), n_width - 1)
    row = min(int(py_norm * n_height), n_height - 1)
    return (col / n_width, row / n_height, (col + 1) / n_width, (row + 1) / n_height)


def point_in_bbox(px, py, bbox):
    """Check if point (px, py) falls within bbox (x1, y1, x2, y2)."""
    return bbox[0] <= px <= bbox[2] and bbox[1] <= py <= bbox[3]


def precision_for_round(round_idx):
    """Return max_pixels for a given 0-based round index (same as training)."""
    return PRECISION_LOW if round_idx == 0 else PRECISION_HIGH


# ---------------------------------------------------------------------------
# Attention helpers (from training code)
# ---------------------------------------------------------------------------

def _find_nth_image_visual_range(input_ids, image_token_id, n):
    """Return (start, end) indices of the n-th contiguous block of image tokens."""
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
    return blocks[n] if n < len(blocks) else None


def _find_nth_pointer_pad(input_ids, pointer_pad_id, n):
    """Return index of the n-th pointer_pad token."""
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


def get_round_attention(model, input_ids, pixel_values, image_grid_thw,
                        attention_mask, round_idx):
    """
    Forward-pass the full multi-round sequence and extract attention
    for a specific round's pointer_pad over that round's visual tokens.
    (Identical to training code)
    """
    device = input_ids.device
    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id
    ps_id = model.config.pointer_start_token_id

    vis_range = _find_nth_image_visual_range(input_ids[0], img_tok, round_idx)
    if vis_range is None:
        return None
    vis_start, vis_end = vis_range
    visual_indices = torch.arange(vis_start, vis_end, device=device)

    target_pos = _find_nth_pointer_pad(input_ids[0], pp_id, round_idx)
    if target_pos is None:
        return None
    target_indices = torch.tensor([target_pos], device=device)

    ps_positions = (input_ids[0] == ps_id).nonzero(as_tuple=False).squeeze(-1)
    ps_before = ps_positions[ps_positions < target_pos]
    query_end = ps_before[-1].item() if len(ps_before) > 0 else target_pos
    query_start = vis_end

    query_indices = torch.arange(query_start, query_end, device=device)
    if getattr(model.config, 'part_query_weighting', False) and len(query_indices) > 12:
        query_indices = query_indices[:-12]
    if query_indices.numel() == 0:
        query_indices = torch.arange(max(0, target_pos - 10), target_pos, device=device)

    merged_indices = torch.cat([query_indices, target_indices], dim=0)

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

    all_layer_hs = torch.stack(hs_per_layer[1:], dim=0)
    sample_hs = all_layer_hs[:, 0, :, :]
    q_hs = F.normalize(sample_hs[:, query_indices, :], dim=-1)
    v_hs = F.normalize(sample_hs[:, visual_indices, :], dim=-1)
    sim = torch.einsum('lqd,lvd->lqv', q_hs, v_hs)

    topk_query_indices = None
    global_pattern = None
    if not getattr(model.config, 'kl_query_weighting', False):
        k = getattr(model.config, 'query_topk', 1)
        agg = sim.sum(dim=-1).sum(dim=0)
        _, topk_query_indices = torch.topk(agg, min(k, len(query_indices)), largest=True)
    else:
        global_pattern = sim.sum(dim=-1).sum(dim=0).softmax(dim=-1)

    attn_weights, _ = model.multi_patch_pointer_head_attention(
        query_indices, visual_indices, target_indices,
        calculated_attention[0],
        topk_query_indices, global_pattern,
        batch_idx=0,
    )

    merge = model.visual.spatial_merge_size
    if round_idx < image_grid_thw.shape[0]:
        _, nh, nw = (image_grid_thw[round_idx] // merge).tolist()
    else:
        n_vis = visual_indices.numel()
        nw = nh = int(math.sqrt(n_vis))

    return {"attn_weights": attn_weights, "n_width": int(nw), "n_height": int(nh)}


def argmax_from_attention(attn_weights, n_w, n_h):
    """Deterministic prediction: argmax of attention distribution."""
    p = attn_weights.squeeze(0).float()
    idx = p.argmax().item()
    px = idx % n_w
    py = idx // n_w
    return (px + 0.5) / n_w, (py + 0.5) / n_h


# ---------------------------------------------------------------------------
# MultiRoundInputBuilder (from training code, adapted for eval)
# ---------------------------------------------------------------------------

class MultiRoundInputBuilder:
    """Incrementally builds the multi-round conversation and tokenises it.
    Identical to training code's builder."""

    def __init__(self, model_path, tokenizer, min_pixels):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.min_pixels = min_pixels
        self._processor_cache = {}

    def _get_processor(self, max_pixels):
        if max_pixels not in self._processor_cache:
            p = AutoProcessor.from_pretrained(
                self.model_path, min_pixels=self.min_pixels, max_pixels=max_pixels,
            )
            p.tokenizer = self.tokenizer
            self._processor_cache[max_pixels] = p
        return self._processor_cache[max_pixels]

    def build_round0(self, image_path, instruction, max_pixels):
        """Build round-0 inputs (full image)."""
        conv = [
            {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction},
            ]},
        ]
        text = self._get_processor(max_pixels).apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False, chat_template=chat_template,
        )
        text += PLACEHOLDER_SUFFIX
        images, _ = process_vision_info(conv)
        inputs = self._get_processor(max_pixels)(text=[text], images=images, return_tensors="pt", padding=True)
        return inputs, text, images

    def extend_with_crop(self, prev_text, prev_images, crop_pil, crop_bbox, round_idx):
        """Append a crop round (identical to training)."""
        max_px = precision_for_round(round_idx)
        zoom_text = (
            f"\n<|im_start|>user\n<image>"
            f"[Zoomed region round {round_idx + 1} around "
            f"({crop_bbox[0]:.2f},{crop_bbox[1]:.2f})-({crop_bbox[2]:.2f},{crop_bbox[3]:.2f})]"
            f"<|im_end|>\n"
            + PLACEHOLDER_SUFFIX
        )
        new_text = prev_text + zoom_text
        new_images = prev_images + [crop_pil]
        proc = self._get_processor(max_px)
        inputs = proc(text=[new_text], images=new_images, return_tensors="pt", padding=True)
        return inputs, new_text, new_images


# ---------------------------------------------------------------------------
# Normalize bbox helper
# ---------------------------------------------------------------------------

def normalize_bbox(bbox_x1y1x2y2, img_width, img_height):
    x1, y1, x2, y2 = bbox_x1y1x2y2
    if (0 <= x1 <= 1) and (0 <= y1 <= 1) and (0 <= x2 <= 1) and (0 <= y2 <= 1):
        return bbox_x1y1x2y2
    return (x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height)


# ---------------------------------------------------------------------------
# Metrics (from original eval)
# ---------------------------------------------------------------------------

def get_metric(list_of_examples,
               groups=["Dev", "Creative", "CAD", "Scientific", "Office", "OS"],
               ui_types=["text", "icon"]):
    metrics = ["hit_top1", "overlap_top1"]

    def compute_mean(examples, key):
        if not examples:
            return None
        return sum(example.get(key, 0) for example in examples) / len(examples)

    results = {metric: {} for metric in metrics}

    for group in groups:
        group_examples = [ex for ex in list_of_examples if ex.get("group") == group]
        for ui in ui_types:
            group_ui_examples = [ex for ex in group_examples if ex.get("ui_type") == ui]
            col_name = f"{group}-{ui}"
            for metric in metrics:
                results[metric][col_name] = compute_mean(group_ui_examples, metric)
        col_name_avg = f"{group}-avg"
        for metric in metrics:
            results[metric][col_name_avg] = compute_mean(group_examples, metric)

    for ui in ui_types:
        ui_examples = [ex for ex in list_of_examples if ex.get("ui_type") == ui]
        col_name = f"All-{ui}"
        for metric in metrics:
            results[metric][col_name] = compute_mean(ui_examples, metric)

    overall_key = "All-avg"
    for metric in metrics:
        results[metric][overall_key] = compute_mean(list_of_examples, metric)

    columns_order = []
    for group in groups:
        for ui in ui_types:
            columns_order.append(f"{group}-{ui}")
        columns_order.append(f"{group}-avg")
    for ui in ui_types:
        columns_order.append(f"All-{ui}")
    columns_order.append("All-avg")

    header = [""] + columns_order
    col_widths = [max(len(col), 12) for col in header]

    def format_cell(cell):
        if isinstance(cell, float):
            return f"{cell * 100:.2f}"
        elif cell is None:
            return "N/A"
        return str(cell)

    header_line = " | ".join(word.ljust(width) for word, width in zip(header, col_widths))
    separator_line = "-+-".join("-" * width for width in col_widths)
    print(header_line)
    print(separator_line)

    for metric in metrics:
        row = [metric]
        for col in columns_order:
            val = results[metric].get(col)
            row.append(format_cell(val))
        row_line = " | ".join(word.ljust(width) for word, width in zip(row, col_widths))
        print(row_line)

    metric_info = "Tab-delimited Table for Excel:\n"
    header_tab = "\t".join([""] + columns_order)
    metric_info += header_tab + "\n"
    for metric in metrics:
        row = [metric] + [format_cell(results[metric].get(col)) for col in columns_order]
        metric_info += ("\t".join(row) + "\n")
    print(metric_info)
    return metric_info


# ---------------------------------------------------------------------------
# Single-round inference (GUI-AIMA standard, for comparison)
# ---------------------------------------------------------------------------

def run_single_round_standard(image, instruction, model, tokenizer, processor,
                              use_placeholder=True, topk=3, resize_to_pixels=None):
    """Single-round using GUI-AIMA's inference() function."""
    if resize_to_pixels is not None and resize_to_pixels > 0:
        w, h = image.size
        if w * h != resize_to_pixels:
            ratio = (resize_to_pixels / (w * h)) ** 0.5
            image = image.resize((int(w * ratio), int(h * ratio)))

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": GROUNDING_SYSTEM_MESSAGE}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ]},
    ]
    pred = inference(conversation, model, tokenizer, processor,
                     logits_processor=None, use_placeholder=use_placeholder, topk=topk)
    return pred


# ---------------------------------------------------------------------------
# Multi-round attention-based inference (ALIGNED with training)
# ---------------------------------------------------------------------------

def run_multi_round_aligned(image, image_path, instruction, model, tokenizer,
                            builder, max_rounds=5, crop_ratio=0.3, device="cuda:0"):
    """
    Multi-round inference using attention argmax, fully aligned with training.
    
    Key differences from training:
    - Uses argmax instead of sampling (deterministic eval)
    - Same input construction (MultiRoundInputBuilder)
    - Same convergence criterion (point_in_bbox, ri >= 2)
    - Same precision levels
    """
    # Round 0: full image, low resolution
    r0_inputs, r0_text, r0_images = builder.build_round0(
        image_path, instruction, precision_for_round(0)
    )
    r0_dev = {k: v.to(device) for k, v in r0_inputs.items()}

    attn0 = get_round_attention(
        model, r0_dev["input_ids"], r0_dev.get("pixel_values"),
        r0_dev.get("image_grid_thw"),
        r0_dev.get("attention_mask", torch.ones_like(r0_dev["input_ids"])),
        round_idx=0,
    )
    if attn0 is None:
        return {"topk_points": [(0.5, 0.5)], "n_width": 1, "n_height": 1, "num_rounds": 0}

    # Argmax prediction (deterministic, vs sampling in training)
    px, py = argmax_from_attention(attn0["attn_weights"], attn0["n_width"], attn0["n_height"])

    round_coords = [(px, py)]
    round_local_coords = [(px, py)]
    round_crop_bboxes = [None]  # round 0 = full image

    prev_x, prev_y = px, py
    prev_nw, prev_nh = attn0["n_width"], attn0["n_height"]
    cur_text, cur_images = r0_text, list(r0_images)

    if max_rounds <= 1:
        return {
            "topk_points": [(px, py)],
            "n_width": attn0["n_width"],
            "n_height": attn0["n_height"],
            "num_rounds": 1,
        }

    for ri in range(1, max_rounds):
        cropped, cbbox = crop_image(image, prev_x, prev_y, crop_ratio)
        try:
            ri_inputs, cur_text, cur_images = builder.extend_with_crop(
                cur_text, cur_images, cropped, cbbox, ri,
            )
        except Exception as e:
            print(f"  Round {ri + 1} input build error: {e}")
            break

        ri_dev = {k: v.to(device) for k, v in ri_inputs.items()}
        attn_ri = get_round_attention(
            model, ri_dev["input_ids"], ri_dev.get("pixel_values"),
            ri_dev.get("image_grid_thw"),
            ri_dev.get("attention_mask", torch.ones_like(ri_dev["input_ids"])),
            round_idx=ri,
        )
        if attn_ri is None:
            break

        # Argmax prediction for this round
        cx, cy = argmax_from_attention(attn_ri["attn_weights"], attn_ri["n_width"], attn_ri["n_height"])

        # Map back to original coords
        bx1, by1, bx2, by2 = cbbox
        ox = bx1 + cx * (bx2 - bx1)
        oy = by1 + cy * (by2 - by1)

        round_coords.append((ox, oy))
        round_local_coords.append((cx, cy))
        round_crop_bboxes.append(cbbox)

        # Convergence check (IDENTICAL to training):
        # Does the new prediction fall within the previous round's predicted patch?
        prev_patch_local = get_patch_bbox(
            round_local_coords[-2][0], round_local_coords[-2][1],
            prev_nw, prev_nh,
        )
        if round_crop_bboxes[-2] is not None:
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
            prev_patch_orig = prev_patch_local

        # Only allow convergence from round 2+ (same as training)
        if ri >= 2 and point_in_bbox(ox, oy, prev_patch_orig):
            break

        prev_x, prev_y = ox, oy
        prev_nw, prev_nh = attn_ri["n_width"], attn_ri["n_height"]

    final_x, final_y = round_coords[-1]
    return {
        "topk_points": [(final_x, final_y)],
        "n_width": prev_nw,
        "n_height": prev_nh,
        "num_rounds": len(round_coords),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all(model, tokenizer, processor, data, image_dir, args, builder):
    """Run evaluation over all examples."""
    device = next(model.parameters()).device
    results = []

    for i, example in tqdm(enumerate(data), total=len(data)):
        ele = {
            "file_name": example["img_filename"],
            "ui_type": example["ui_type"],
            "group": example["group"],
            "platform": example["platform"],
            "application": example["application"],
            "id": example["id"],
            "instruction": example["instruction"],
            "img_size": example["img_size"],
            "bbox_x1y1x2y2": normalize_bbox(example["bbox"], example["img_size"][0], example["img_size"][1]),
            "hit_top1": 0,
            "overlap_top1": 0,
        }

        image_path = os.path.join(image_dir, example["img_filename"])
        image = Image.open(image_path).convert("RGB")
        gt_bbox = ele["bbox_x1y1x2y2"]

        if args.mode == "standard":
            # Standard GUI-AIMA single-round (for baseline comparison)
            pred = run_single_round_standard(
                image, example["instruction"], model, tokenizer, processor,
                use_placeholder=True, topk=args.topk,
                resize_to_pixels=args.resize_to_pixels if args.resize_to_pixels > 0 else None,
            )
        elif args.mode == "aligned":
            # Multi-round attention-based (aligned with training)
            pred = run_multi_round_aligned(
                image, image_path, example["instruction"],
                model, tokenizer, builder,
                max_rounds=args.rounds, crop_ratio=args.crop_ratio,
                device=str(device),
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        topk_points = pred["topk_points"]
        if not topk_points:
            results.append(ele)
            continue

        n_w = pred.get("n_width", 1)
        n_h = pred.get("n_height", 1)
        IMAGE_PATCH_SIZE_x = 0.5 / max(n_w, 1)
        IMAGE_PATCH_SIZE_y = 0.5 / max(n_h, 1)

        x1, y1, x2, y2 = gt_bbox
        px, py = topk_points[0]

        if (x1 <= px <= x2) and (y1 <= py <= y2):
            ele["hit_top1"] = 1

        pred_bbox = [px - IMAGE_PATCH_SIZE_x, py - IMAGE_PATCH_SIZE_y,
                     px + IMAGE_PATCH_SIZE_x, py + IMAGE_PATCH_SIZE_y]
        if do_boxes_overlap(pred_bbox, gt_bbox):
            ele["overlap_top1"] = 1

        ele["pred_x"] = px
        ele["pred_y"] = py
        ele["num_rounds"] = pred.get("num_rounds", 1)

        results.append(ele)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ScreenSpot-Pro eval (aligned with training)")

    # Model
    parser.add_argument("--model_name_or_path", type=str, default="smz8599/GUI-AIMA-3B")

    # Data
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/data/ScreenSpot-Pro")
    parser.add_argument("--save_path", type=str, default=None)

    # Mode
    parser.add_argument("--mode", type=str, default="aligned",
                        choices=["standard", "aligned"],
                        help="'standard' = GUI-AIMA single-round, 'aligned' = multi-round attention-based")

    # Resolution (for standard mode)
    parser.add_argument("--resolution", type=str, default="high", choices=["high", "low"])
    parser.add_argument("--resize_to_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)

    # Multi-round (for aligned mode)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--crop_ratio", type=float, default=0.3)

    # Misc
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    # Resolve resolution
    if args.mode == "standard":
        if args.resolution == "high":
            args.resize_to_pixels = args.resize_to_pixels or 5760000
            args.max_pixels = args.max_pixels or 14777616
        else:
            args.resize_to_pixels = args.resize_to_pixels or -1
            args.max_pixels = args.max_pixels or 1003520
    else:
        # For aligned mode, we use PRECISION_HIGH as max_pixels for processor
        args.max_pixels = args.max_pixels or 14777616
        args.resize_to_pixels = args.resize_to_pixels or -1

    # Auto save path
    if args.save_path is None:
        model_name = Path(args.model_name_or_path).name
        if args.mode == "aligned":
            tag = f"aligned_r{args.rounds}_crop{args.crop_ratio}"
        else:
            tag = f"standard_{args.resolution}"
        args.save_path = f"/root/autodl-tmp/results/screenspot_pro/{model_name}/{tag}"

    print(f"=== Config ===")
    print(f"  mode:             {args.mode}")
    print(f"  rounds:           {args.rounds}")
    print(f"  crop_ratio:       {args.crop_ratio}")
    print(f"  max_pixels:       {args.max_pixels}")
    print(f"  device:           {args.device}")
    print()

    # Load data
    data_fn = os.path.join(args.data_path, "annotations/all.json")
    with open(data_fn, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {data_fn}")

    if args.max_samples is not None:
        data = data[:args.max_samples]
        print(f"Limiting to {len(data)} samples")

    image_dir = os.path.join(args.data_path, "images")

    # Load model
    print(f"Loading model: {args.model_name_or_path} (max_pixels={args.max_pixels})")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, max_pixels=args.max_pixels)
    tokenizer = processor.tokenizer

    # Add special tokens if not present
    num_new = tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    if num_new > 0:
        print(f"Added {num_new} special tokens to tokenizer")

    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
    ).eval()

    if num_new > 0:
        model.resize_token_embeddings(len(tokenizer))

    for attr, default in [
        ('query_topk', None), ('kl_query_weighting', False),
        ('part_query_weighting', False), ('layer_wise_query_weighting', False),
    ]:
        if not hasattr(model.config, attr):
            setattr(model.config, attr, default)

    # Ensure pointer token IDs are set (needed for attention extraction)
    if not hasattr(model.config, 'pointer_pad_token_id') or model.config.pointer_pad_token_id is None:
        model.config.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
        model.config.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        model.config.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
        print(f"Set pointer token IDs: start={model.config.pointer_start_token_id}, "
              f"pad={model.config.pointer_pad_token_id}, end={model.config.pointer_end_token_id}")

    # Builder for aligned mode
    builder = None
    if args.mode == "aligned":
        builder = MultiRoundInputBuilder(args.model_name_or_path, tokenizer, min_pixels=3136)

    # Check cache
    os.makedirs(args.save_path, exist_ok=True)
    model_name = Path(args.model_name_or_path).name
    pred_path = os.path.join(args.save_path, f"{model_name}_preds.json")
    metric_path = os.path.join(args.save_path, f"{model_name}_metric.txt")

    if os.path.exists(pred_path):
        print(f"Loading cached predictions from {pred_path}")
        with open(pred_path, "r") as f:
            results = json.load(f)
    else:
        t0 = time.time()
        with torch.no_grad():
            results = evaluate_all(model, tokenizer, processor, data, image_dir, args, builder)
        elapsed = time.time() - t0
        print(f"Evaluation took {elapsed:.1f}s ({elapsed / len(results):.2f}s/sample)")

        with open(pred_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} predictions to {pred_path}")

        # Round statistics
        if args.mode == "aligned":
            round_counts = {}
            for r in results:
                nr = r.get("num_rounds", 1)
                round_counts[nr] = round_counts.get(nr, 0) + 1
            print(f"\nRound distribution: {dict(sorted(round_counts.items()))}")

    # Compute metrics
    if not os.path.exists(metric_path):
        print(f"\n=== Metrics ===")
        metric_info = get_metric(results)
        with open(metric_path, "w") as f:
            f.write(metric_info)
        print(f"Saved metrics to {metric_path}")
    else:
        print(f"Metrics already exist at {metric_path}")
        with open(metric_path, "r") as f:
            print(f.read())


if __name__ == "__main__":
    main()
