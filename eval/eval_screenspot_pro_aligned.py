"""
ScreenSpot-Pro evaluation script ALIGNED with multi-round GRPO training.

Key alignment with gui_attention.train:
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
import os
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.inference import inference
from gui_aima.utils import do_boxes_overlap
from gui_aima.constants import (
    ADDITIONAL_SPECIAL_TOKENS,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    grounding_system_message,
)
from transformers import AutoProcessor, AutoTokenizer

from gui_attention.constants import precision_for_round
from gui_attention.crop import crop_image, get_patch_bbox, point_in_bbox
from gui_attention.attention import get_round_attention
from gui_attention.sampling import argmax_from_attention, region_from_attention
from gui_attention.builder import MultiRoundInputBuilder


# ---------------------------------------------------------------------------
# Helpers
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
        {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
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
                            builder, max_rounds=5, crop_ratio=0.3, device="cuda:0",
                            prediction_method="region"):
    """
    Multi-round inference using attention-based prediction, aligned with training.

    Key differences from training:
    - Uses deterministic prediction instead of sampling
    - Same input construction (MultiRoundInputBuilder)
    - Same convergence criterion (point_in_bbox, ri >= 2)
    - Same precision levels
    """
    predict_fn = region_from_attention if prediction_method == "region" else argmax_from_attention
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

    # Deterministic prediction (vs sampling in training)
    px, py = predict_fn(attn0["attn_weights"], attn0["n_width"], attn0["n_height"])

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

        # Deterministic prediction for this round
        cx, cy = predict_fn(attn_ri["attn_weights"], attn_ri["n_width"], attn_ri["n_height"])

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
                prediction_method=args.prediction_method,
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

    # Prediction method
    parser.add_argument("--prediction_method", type=str, default="region",
                        choices=["argmax", "region"],
                        help="'argmax' = naive argmax, 'region' = GUI-AIMA region-based (recommended)")

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
            tag = f"aligned_r{args.rounds}_crop{args.crop_ratio}_{args.prediction_method}"
        else:
            tag = f"standard_{args.resolution}"
        args.save_path = f"/root/autodl-tmp/results/screenspot_pro/{model_name}/{tag}"

    print(f"=== Config ===")
    print(f"  mode:             {args.mode}")
    print(f"  prediction:       {args.prediction_method}")
    print(f"  rounds:           {args.rounds}")
    print(f"  crop_ratio:       {args.crop_ratio}")
    print(f"  max_pixels:       {args.max_pixels}")
    # Force cuda:0 for aligned mode — device_map="auto" breaks output_hidden_states
    if args.mode == "aligned" and args.device == "auto":
        args.device = "cuda:0"
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
        ('query_topk', 1), ('kl_query_weighting', False),
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
