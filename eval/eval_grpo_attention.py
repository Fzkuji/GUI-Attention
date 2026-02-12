"""
Evaluate GRPO attention-sampling trained models on ScreenSpot-Pro.

Supports two modes:
1. Standard (argmax): Use GUI-AIMA's standard attention argmax for prediction
2. Sampling: Sample from attention distribution (with low temperature for eval)

Reports accuracy (bbox hit rate) and average distance.
"""

import argparse
import json
import os
import sys
import glob
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, "/root/GUI-AIMA/src")

from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.inference import inference as guiaima_inference
from gui_aima.constants import (
    chat_template, grounding_system_message,
    ADDITIONAL_SPECIAL_TOKENS, DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN, DEFAULT_POINTER_PAD_TOKEN,
)
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info


def crop_image(image, center_x, center_y, crop_ratio):
    """Crop a region around (center_x, center_y) in normalized [0,1] coords."""
    W, H = image.size
    crop_w = int(W * crop_ratio)
    crop_h = int(H * crop_ratio)
    cx_px = int(center_x * W)
    cy_px = int(center_y * H)
    x1 = max(0, cx_px - crop_w // 2)
    y1 = max(0, cy_px - crop_h // 2)
    x2 = min(W, x1 + crop_w)
    y2 = min(H, y1 + crop_h)
    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)
    cropped = image.crop((x1, y1, x2, y2))
    bbox = (x1 / W, y1 / H, x2 / W, y2 / H)
    return cropped, bbox


def load_screenspot_pro(data_path):
    """Load ScreenSpot-Pro annotations."""
    annotations_dir = os.path.join(data_path, "annotations")
    data = []
    for f in sorted(glob.glob(os.path.join(annotations_dir, "*.json"))):
        with open(f) as fh:
            data.extend(json.load(fh))
    return data


def normalize_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    if any(v > 1 for v in [x1, y1, x2, y2]):
        return x1 / w, y1 / h, x2 / w, y2 / h
    return x1, y1, x2, y2


class GRPOAttentionEvaluator:
    """Evaluate a GRPO attention-trained model."""

    def __init__(self, model_path, device="cuda", max_pixels=1003520):
        print(f"Loading model from {model_path}...")
        self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            ignore_mismatched_sizes=True,
        ).to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(
            model_path, max_pixels=max_pixels,
        )
        self.processor.tokenizer = self.tokenizer

        # Ensure special tokens
        num_new = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS}
        )
        if num_new > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Ensure config has pointer IDs
        if not hasattr(self.model.config, 'pointer_pad_token_id') or self.model.config.pointer_pad_token_id is None:
            self.model.config.pointer_start_token_id = self.tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
            self.model.config.pointer_end_token_id = self.tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
            self.model.config.pointer_pad_token_id = self.tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]

        for attr, default in [
            ('query_topk', 1), ('kl_query_weighting', False),
            ('part_query_weighting', False), ('layer_wise_query_weighting', False),
        ]:
            if not hasattr(self.model.config, attr):
                setattr(self.model.config, attr, default)

        self.device = device
        print(f"Model loaded. pointer_pad_token_id={self.model.config.pointer_pad_token_id}")

    @torch.no_grad()
    def predict(self, image, instruction, verbose=False):
        """Standard argmax prediction using GUI-AIMA inference."""
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ]},
        ]

        pred = guiaima_inference(
            conversation, self.model, self.tokenizer, self.processor,
            use_placeholder=True, topk=3,
        )

        if pred["topk_points"] and len(pred["topk_points"]) > 0:
            return pred["topk_points"][0]
        return 0.5, 0.5

    @torch.no_grad()
    def predict_multi_round(self, image, instruction, max_rounds=5,
                            crop_ratio=0.3, convergence_threshold=0.02,
                            fovea_max_pixels=56*56*28*28, verbose=False):
        """
        Multi-round foveation prediction (argmax mode for eval).
        Round 1: low-res full image → predict
        Round 2+: crop around prediction → predict on crop+original → refine
        """
        import math

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Round 1: standard prediction
        px, py = self.predict(image, instruction, verbose=verbose)
        if verbose:
            print(f"  R1: pred=({px:.4f}, {py:.4f})")

        prev_x, prev_y = px, py

        for round_idx in range(1, max_rounds):
            # Crop around previous prediction
            cropped, crop_bbox = crop_image(image, prev_x, prev_y, crop_ratio)

            # Build multi-image conversation
            content_items = [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
                {"type": "image", "image": cropped},
                {"type": "text", "text": f"[Zoomed region around ({crop_bbox[0]:.2f},{crop_bbox[1]:.2f})-({crop_bbox[2]:.2f},{crop_bbox[3]:.2f})]"},
            ]
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
                {"role": "user", "content": content_items},
            ]

            try:
                pred = guiaima_inference(
                    conversation, self.model, self.tokenizer, self.processor,
                    use_placeholder=True, topk=3,
                )
            except Exception as e:
                if verbose:
                    print(f"  R{round_idx+1}: error - {e}")
                break

            if pred["topk_points"] and len(pred["topk_points"]) > 0:
                local_x, local_y = pred["topk_points"][0]
                # Map crop-local coords back to original
                bx1, by1, bx2, by2 = crop_bbox
                cur_x = bx1 + local_x * (bx2 - bx1)
                cur_y = by1 + local_y * (by2 - by1)
            else:
                break

            if verbose:
                print(f"  R{round_idx+1}: pred=({cur_x:.4f}, {cur_y:.4f})")

            # Check convergence
            dist = math.sqrt((cur_x - prev_x) ** 2 + (cur_y - prev_y) ** 2)
            prev_x, prev_y = cur_x, cur_y

            if dist < convergence_threshold:
                if verbose:
                    print(f"  Converged at round {round_idx+1}")
                break

        return prev_x, prev_y


def evaluate(evaluator, data, image_dir, n_samples=None, verbose=False,
             multi_round=False, max_rounds=5, crop_ratio=0.3, convergence_threshold=0.02):
    results = []
    samples = data[:n_samples] if n_samples else data

    for i, ex in tqdm(enumerate(samples), total=len(samples)):
        img_path = os.path.join(image_dir, ex["img_filename"])
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        iw, ih = ex["img_size"]
        bbox = normalize_bbox(ex["bbox"], iw, ih)
        x1, y1, x2, y2 = bbox
        gt_cx, gt_cy = (x1 + x2) / 2, (y1 + y2) / 2

        try:
            if multi_round:
                px, py = evaluator.predict_multi_round(
                    image, ex["instruction"],
                    max_rounds=max_rounds, crop_ratio=crop_ratio,
                    convergence_threshold=convergence_threshold,
                    verbose=verbose,
                )
            else:
                px, py = evaluator.predict(image, ex["instruction"], verbose=verbose)
        except Exception as e:
            print(f"Error on {i}: {e}")
            px, py = 0.5, 0.5

        hit = 1 if (x1 <= px <= x2 and y1 <= py <= y2) else 0
        dist = ((px - gt_cx) ** 2 + (py - gt_cy) ** 2) ** 0.5

        results.append({
            "id": i,
            "instruction": ex["instruction"],
            "pred": (px, py),
            "gt_bbox": bbox,
            "hit": hit,
            "dist": dist,
        })

        if (i + 1) % 50 == 0:
            acc = sum(r["hit"] for r in results) / len(results) * 100
            avg_dist = sum(r["dist"] for r in results) / len(results)
            print(f"  [{i + 1}/{len(samples)}] acc={acc:.1f}% avg_dist={avg_dist:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to GRPO attention-trained checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to ScreenSpot-Pro dataset")
    parser.add_argument("--save_path", type=str, default="/root/autodl-tmp/results/grpo_attention")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=1003520)
    parser.add_argument("--multi_round", action="store_true",
                        help="Enable multi-round foveation evaluation")
    parser.add_argument("--max_rounds", type=int, default=5)
    parser.add_argument("--crop_ratio", type=float, default=0.3)
    parser.add_argument("--convergence_threshold", type=float, default=0.02)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data = load_screenspot_pro(args.data_path)
    image_dir = os.path.join(args.data_path, "images")
    print(f"Loaded {len(data)} samples")

    evaluator = GRPOAttentionEvaluator(
        args.model_path, max_pixels=args.max_pixels,
    )

    mode_str = f"Multi-Round ({args.max_rounds} rounds)" if args.multi_round else "Single-Round"
    print(f"\n=== Evaluating GRPO Attention Model ({mode_str}) ===")
    t0 = time.time()
    results = evaluate(
        evaluator, data, image_dir, args.n_samples, args.verbose,
        multi_round=args.multi_round, max_rounds=args.max_rounds,
        crop_ratio=args.crop_ratio, convergence_threshold=args.convergence_threshold,
    )
    elapsed = time.time() - t0

    n = len(results)
    acc = sum(r["hit"] for r in results) / n * 100
    avg_dist = sum(r["dist"] for r in results) / n

    print(f"\n{'=' * 60}")
    print(f"GRPO Attention Model: {sum(r['hit'] for r in results)}/{n} ({acc:.1f}%)")
    print(f"Average distance: {avg_dist:.4f}")
    print(f"Time: {elapsed:.1f}s ({elapsed / n:.2f}s/sample)")
    print(f"{'=' * 60}")

    os.makedirs(args.save_path, exist_ok=True)
    out_file = os.path.join(args.save_path, "grpo_attention_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "accuracy": acc,
            "avg_dist": avg_dist,
            "n_samples": n,
            "time_seconds": elapsed,
            "model_path": args.model_path,
            "results": [(r["hit"], r["dist"]) for r in results],
        }, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
