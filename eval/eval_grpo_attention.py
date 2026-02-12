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


def evaluate(evaluator, data, image_dir, n_samples=None, verbose=False):
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    data = load_screenspot_pro(args.data_path)
    image_dir = os.path.join(args.data_path, "images")
    print(f"Loaded {len(data)} samples")

    evaluator = GRPOAttentionEvaluator(
        args.model_path, max_pixels=args.max_pixels,
    )

    print(f"\n=== Evaluating GRPO Attention Model ===")
    t0 = time.time()
    results = evaluate(evaluator, data, image_dir, args.n_samples, args.verbose)
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
