"""
ScreenSpot-Pro evaluation using multi-precision foveation (v3).

Uses gui_aima's multi-layer multi-head attention aggregation with
visual-sink head weighting and FoveationLoop for progressive zoom.

Usage:
  # Multi-round foveation eval
  python eval/eval_screenspot_pro_aligned.py \
      --model_name_or_path /path/to/checkpoint \
      --rounds 5 --crop_ratio 0.3

  # Single-round eval at specific precision level
  python eval/eval_screenspot_pro_aligned.py \
      --model_name_or_path /path/to/checkpoint \
      --rounds 1 --initial_level 1
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.constants import ADDITIONAL_SPECIAL_TOKENS
from transformers import AutoProcessor, AutoTokenizer

from gui_attention.constants import (
    precision_for_level,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
)
from gui_attention.crop import crop_image
from gui_attention.attention import (
    find_image_visual_ranges,
    find_nth_pointer_pad,
    extract_attention,
    identify_attended_image,
    token_to_spatial,
)
from gui_attention.builder import MultiRoundInputBuilder
from gui_attention.foveation import FoveationLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def do_boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    return not (box1[2] < box2[0] or box1[0] > box2[2] or
                box1[3] < box2[1] or box1[1] > box2[3])


def normalize_bbox(bbox_x1y1x2y2, img_width, img_height):
    x1, y1, x2, y2 = bbox_x1y1x2y2
    if (0 <= x1 <= 1) and (0 <= y1 <= 1) and (0 <= x2 <= 1) and (0 <= y2 <= 1):
        return bbox_x1y1x2y2
    return (x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height)


def _get_all_visual_indices(input_ids, image_token_id, up_to_pos=None):
    """Get indices of ALL visual tokens, optionally only those before a given position."""
    ranges = find_image_visual_ranges(input_ids, image_token_id)
    indices = []
    range_offsets = []
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


def _get_query_indices(input_ids, image_token_id, pointer_pad_id, up_to_pos):
    """Get indices of text tokens before up_to_pos."""
    vis_set = set()
    ranges = find_image_visual_ranges(input_ids, image_token_id)
    for vs, ve in ranges:
        for i in range(vs, ve):
            vis_set.add(i)
    pp_set = set(pointer_pad_id) if isinstance(pointer_pad_id, list) else {pointer_pad_id}
    query_indices = []
    for i in range(up_to_pos):
        if i not in vis_set and input_ids[i].item() not in pp_set:
            query_indices.append(i)
    return query_indices


# ---------------------------------------------------------------------------
# Metrics
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
# Multi-round foveation inference (v3)
# ---------------------------------------------------------------------------

def run_multi_round_foveation(image, image_path, instruction, model, tokenizer,
                               builder, max_rounds=5, crop_ratio=0.3,
                               device="cuda:0", initial_level=0):
    """
    Multi-round inference using multi-precision foveation with gui_aima attention.
    """
    fov_loop = FoveationLoop(max_rounds=max_rounds, crop_ratio=crop_ratio)
    state = fov_loop.new_state()

    builder.reset()
    level = initial_level

    r0_inputs, cur_text, cur_images = builder.build_round0(
        image_path, instruction, level=level,
    )
    last_inputs = r0_inputs

    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id
    merge = model.visual.spatial_merge_size

    round_coords = []
    nw, nh = 1, 1

    for ri in range(max_rounds):
        inp = {k: v.to(device) for k, v in last_inputs.items()}
        input_ids = inp["input_ids"]
        attention_mask = inp.get("attention_mask", torch.ones_like(input_ids))

        # Get position IDs
        position_ids, _ = model.get_rope_index(
            input_ids=input_ids, image_grid_thw=inp.get("image_grid_thw"),
            video_grid_thw=None, attention_mask=attention_mask,
        )

        # Forward pass
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=inp.get("pixel_values"),
            image_grid_thw=inp.get("image_grid_thw"),
            output_hidden_states=True,
        )

        # Find pointer_pad for this round
        ptr_pos = find_nth_pointer_pad(input_ids[0], pp_id, ri)
        if ptr_pos is None:
            break

        # Visual and query indices
        visual_indices, range_offsets = _get_all_visual_indices(
            input_ids[0], img_tok, up_to_pos=ptr_pos,
        )
        if visual_indices is None:
            break

        query_indices = _get_query_indices(
            input_ids[0], img_tok, pp_id, up_to_pos=ptr_pos,
        )

        # Extract multi-layer attention
        attn_weights, _ = extract_attention(
            model, outputs, input_ids, position_ids, attention_mask,
            visual_indices=visual_indices.tolist(),
            query_indices=query_indices,
            target_index=ptr_pos,
        )

        # Identify which image has the max-attended token
        attn_1d = attn_weights.squeeze(0)
        img_idx, local_idx = identify_attended_image(
            attn_1d, [(0, ro[1]) for ro in range_offsets],
        )

        # Get spatial coordinates
        grid_dims = builder.get_image_grid_dims(inp["image_grid_thw"], merge)
        if img_idx < len(grid_dims):
            nh, nw = grid_dims[img_idx]
        else:
            n_vis = range_offsets[img_idx][1] if img_idx < len(range_offsets) else 1
            nw = nh = int(math.sqrt(n_vis))

        lx, ly = token_to_spatial(local_idx, nw, nh)

        # Map to global coordinates
        info = builder.image_infos[img_idx]
        bx1, by1, bx2, by2 = info.global_bbox
        ox = bx1 + lx * (bx2 - bx1)
        oy = by1 + ly * (by2 - by1)

        attended_level = info.level
        round_coords.append((ox, oy))

        # Foveation decision
        decision = fov_loop.decide(state, attended_level, ox, oy)

        if decision["action"] == "stop":
            break

        if decision["action"] == "crop":
            next_level = decision["level"]
            cropped, cbbox = crop_image(image, ox, oy, crop_ratio)
            try:
                ri_inputs, cur_text, cur_images = builder.extend_with_crop(
                    cur_text, cur_images, cropped, cbbox, level=next_level,
                )
            except Exception as e:
                print(f"  Round {ri + 1} build error: {e}")
                break
            last_inputs = ri_inputs

        if not fov_loop.should_continue(state, ri + 1):
            break

    if not round_coords:
        return {
            "topk_points": [(0.5, 0.5)],
            "n_width": 1,
            "n_height": 1,
            "num_rounds": 0,
        }

    final_x, final_y = round_coords[-1]
    return {
        "topk_points": [(final_x, final_y)],
        "n_width": nw,
        "n_height": nh,
        "num_rounds": len(round_coords),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_all(model, tokenizer, data, image_dir, args, builder):
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

        pred = run_multi_round_foveation(
            image, image_path, example["instruction"],
            model, tokenizer, builder,
            max_rounds=args.rounds, crop_ratio=args.crop_ratio,
            device=str(device), initial_level=args.initial_level,
        )

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
    parser = argparse.ArgumentParser(description="ScreenSpot-Pro eval (multi-precision foveation v3)")

    # Model
    parser.add_argument("--model_name_or_path", type=str, default="smz8599/GUI-AIMA-3B")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Base model path for loading processor (if checkpoint has incompatible processor)")

    # Data
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/data/ScreenSpot-Pro")
    parser.add_argument("--save_path", type=str, default=None)

    # Multi-round foveation
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--crop_ratio", type=float, default=0.3)
    parser.add_argument("--initial_level", type=int, default=0,
                        help="Precision level for round 0 (0=low, 1=original, 2=high, 3=ultra)")

    # Attention config
    parser.add_argument("--query_weighting", type=str, default="query_1",
                        help="Visual-sink weighting strategy (e.g., query_1, query_5, kl)")

    # Misc
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # Auto save path
    if args.save_path is None:
        model_name = Path(args.model_name_or_path).name
        tag = f"fov_r{args.rounds}_L{args.initial_level}_crop{args.crop_ratio}"
        args.save_path = f"/root/autodl-tmp/results/screenspot_pro/{model_name}/{tag}"

    print(f"=== Config ===")
    print(f"  rounds:           {args.rounds}")
    print(f"  initial_level:    {args.initial_level}")
    print(f"  crop_ratio:       {args.crop_ratio}")
    print(f"  query_weighting:  {args.query_weighting}")
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
    max_pixels = precision_for_level(args.initial_level)
    processor_path = args.base_model_path or args.model_name_or_path
    print(f"Loading model: {args.model_name_or_path} (max_pixels={max_pixels})")
    processor = AutoProcessor.from_pretrained(processor_path, max_pixels=max_pixels)
    tokenizer = processor.tokenizer

    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
        ignore_mismatched_sizes=True,
    ).eval()

    # Configure attention args
    model.set_attention_args(args.query_weighting)

    # Ensure pointer token IDs are set
    if not hasattr(model.config, 'pointer_pad_token_id') or model.config.pointer_pad_token_id is None:
        num_new = tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
        if num_new > 0:
            print(f"Added {num_new} special tokens to tokenizer")
            model.resize_token_embeddings(len(tokenizer))
        model.config.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
        model.config.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        model.config.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
        print(f"Set pointer token IDs: start={model.config.pointer_start_token_id}, "
              f"pad={model.config.pointer_pad_token_id}, end={model.config.pointer_end_token_id}")

    # Builder
    builder = MultiRoundInputBuilder(processor_path, tokenizer, min_pixels=3136)

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
            results = evaluate_all(model, tokenizer, data, image_dir, args, builder)
        elapsed = time.time() - t0
        print(f"Evaluation took {elapsed:.1f}s ({elapsed / len(results):.2f}s/sample)")

        with open(pred_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} predictions to {pred_path}")

        # Round statistics
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
