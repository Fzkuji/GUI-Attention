"""
Evaluate baseline models (GUI-AIMA, GUI-Actor) on our training datasets.

Tests how well pretrained baselines already perform on the data we train on,
giving us an upper/lower bound reference.

Supports:
  - GUI-AIMA (pointer head, generation-based attention)
  - GUI-Actor (pointer head, generation-based)
  - Qwen2.5-VL vanilla (text coordinate output)

Usage:
  # GUI-AIMA on GUIAct (sample 1000)
  python eval/eval_baseline_on_trainset.py \
      --model_type gui_aima \
      --model_path /path/to/GUI-AIMA-3B \
      --data_path /path/to/guiact_bbox.json \
      --image_folder /path/to/GUIAct/web_imgs \
      --max_samples 1000 \
      --save_path results/baseline_guiaima_on_guiact

  # Multiple datasets
  python eval/eval_baseline_on_trainset.py \
      --model_type gui_aima \
      --model_path /path/to/GUI-AIMA-3B \
      --data_path d1.json,d2.json \
      --image_folder img1,img2 \
      --max_samples 500

  # GUI-Actor
  python eval/eval_baseline_on_trainset.py \
      --model_type gui_actor \
      --model_path /path/to/GUI-Actor-3B \
      --data_path /path/to/guiact_bbox.json \
      --image_folder /path/to/GUIAct/web_imgs
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# Data loading (same format as gui_attention.train)
# ---------------------------------------------------------------------------

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
            samples.append({
                "image_path": img_path,
                "instruction": user_text,
                "bbox_gt": bbox_gt,  # [x1, y1, x2, y2] normalized 0-1
                "dataset": os.path.basename(data_path),
            })
    return samples


def load_datasets(data_path, image_folder, max_samples=None, max_samples_per_dataset=None, seed=42):
    data_paths = [p.strip() for p in data_path.split(",")]
    image_folders = [p.strip() for p in image_folder.split(",")]
    if len(image_folders) == 1 and len(data_paths) > 1:
        image_folders = image_folders * len(data_paths)
    assert len(data_paths) == len(image_folders)

    per_ds_limits = None
    if max_samples_per_dataset:
        per_ds_limits = [int(x.strip()) for x in max_samples_per_dataset.split(",")]
        assert len(per_ds_limits) == len(data_paths)

    all_samples = []
    for i, (dp, imf) in enumerate(zip(data_paths, image_folders)):
        s = load_single_dataset(dp, imf)
        limit = per_ds_limits[i] if per_ds_limits else 0
        if limit > 0 and len(s) > limit:
            s = s[:limit]
        print(f"  {os.path.basename(dp)}: {len(s)} samples" + (f" (limited to {limit})" if limit > 0 else ""))
        all_samples.extend(s)

    if max_samples and len(all_samples) > max_samples:
        random.seed(seed)
        all_samples = random.sample(all_samples, max_samples)
        print(f"  Sampled {max_samples} from {len(all_samples)} total")

    print(f"Total eval samples: {len(all_samples)}")
    return all_samples


# ---------------------------------------------------------------------------
# GUI-AIMA evaluator
# ---------------------------------------------------------------------------

class GUIAIMAEvaluator:
    def __init__(self, model_path, max_pixels=5720064, device="cuda:0"):
        from transformers import AutoProcessor
        # GUI-AIMA imports
        sys.path.insert(0, str(Path(model_path).parent))  # in case gui_aima is local
        from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
        from gui_aima.inference import inference
        from gui_aima.constants import chat_template

        self.device = device
        self.max_pixels = max_pixels
        self.inference_fn = inference

        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        ).eval()

        self.system_message = (
            "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
            "your task is to locate the screen element that corresponds to the instruction. "
            "You should output a PyAutoGUI action that performs a click on the correct position. "
            "To indicate the click location, we will use some special tokens, which is used to refer "
            "to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
        )

    def predict(self, image, instruction):
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ]},
        ]
        pred = self.inference_fn(
            conversation, self.model, self.tokenizer, self.processor,
            logits_processor=None, use_placeholder=True, topk=1,
        )
        # Returns normalized (0-1) point
        px, py = pred["topk_points"][0]
        return px, py


# ---------------------------------------------------------------------------
# GUI-Actor evaluator
# ---------------------------------------------------------------------------

class GUIActorEvaluator:
    def __init__(self, model_path, max_pixels=5720064, device="cuda:0"):
        from transformers import AutoProcessor
        # GUI-Actor uses same architecture as GUI-AIMA basically
        from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
        from gui_aima.inference import inference

        self.device = device
        self.max_pixels = max_pixels
        self.inference_fn = inference

        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        ).eval()

        self.system_message = (
            "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
            "your task is to locate the screen element that corresponds to the instruction. "
            "You should output a PyAutoGUI action that performs a click on the correct position. "
            "To indicate the click location, we will use some special tokens, which is used to refer "
            "to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
        )

    def predict(self, image, instruction):
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ]},
        ]
        pred = self.inference_fn(
            conversation, self.model, self.tokenizer, self.processor,
            logits_processor=None, use_placeholder=True, topk=1,
        )
        px, py = pred["topk_points"][0]
        return px, py


# ---------------------------------------------------------------------------
# Qwen2.5-VL vanilla (text output) evaluator
# ---------------------------------------------------------------------------

class QwenVanillaEvaluator:
    def __init__(self, model_path, max_pixels=5720064, device="cuda:0"):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = device
        self.max_pixels = max_pixels

        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        ).eval()

    def predict(self, image, instruction):
        """Use Qwen2.5-VL's native grounding: ask for click point, parse output."""
        conversation = [
            {"role": "system", "content": [{"type": "text", "text":
                "You are a GUI grounding agent. Given a screenshot and an instruction, "
                "output the click coordinates as: <points x1=\"X\" y1=\"Y\"> where X,Y are 0-1000."}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ]},
        ]

        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=64)
        output_text = self.tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Parse <points x1="X" y1="Y">
        import re
        m = re.search(r'x1="(\d+)".*?y1="(\d+)"', output_text)
        if m:
            px = int(m.group(1)) / 1000.0
            py = int(m.group(2)) / 1000.0
        else:
            # Try plain numbers
            nums = re.findall(r'(\d+\.?\d*)', output_text)
            if len(nums) >= 2:
                px = float(nums[0]) / 1000.0
                py = float(nums[1]) / 1000.0
            else:
                px, py = 0.5, 0.5  # fallback
        return px, py


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(evaluator, samples, save_path=None):
    hits = 0
    total = 0
    per_dataset = {}  # dataset_name -> {hits, total}
    results = []
    errors = 0

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            errors += 1
            continue

        try:
            px, py = evaluator.predict(image, sample["instruction"])
        except Exception as e:
            print(f"  Inference error: {e}")
            errors += 1
            px, py = 0.5, 0.5

        x1, y1, x2, y2 = sample["bbox_gt"]
        hit = int((x1 <= px <= x2) and (y1 <= py <= y2))
        hits += hit
        total += 1

        ds = sample["dataset"]
        if ds not in per_dataset:
            per_dataset[ds] = {"hits": 0, "total": 0}
        per_dataset[ds]["hits"] += hit
        per_dataset[ds]["total"] += 1

        results.append({
            "image": sample["image_path"],
            "instruction": sample["instruction"],
            "bbox_gt": sample["bbox_gt"],
            "pred": [px, py],
            "hit": hit,
            "dataset": ds,
        })

        # Print running stats every 100 samples
        if total % 100 == 0:
            print(f"  [{total}/{len(samples)}] hit={hits}/{total} ({100*hits/total:.1f}%), errors={errors}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {hits}/{total} = {100*hits/total:.2f}% hit rate")
    print(f"Errors: {errors}")
    print(f"\nPer-dataset breakdown:")
    for ds, stats in sorted(per_dataset.items()):
        h, t = stats["hits"], stats["total"]
        print(f"  {ds}: {h}/{t} = {100*h/t:.2f}%")
    print(f"{'='*60}\n")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        summary = {
            "total": total,
            "hits": hits,
            "hit_rate": hits / total if total else 0,
            "errors": errors,
            "per_dataset": {ds: {**s, "hit_rate": s["hits"]/s["total"]} for ds, s in per_dataset.items()},
        }
        with open(os.path.join(save_path, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(save_path, "details.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models on training datasets")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["gui_aima", "gui_actor", "qwen_vanilla"],
                        help="Which model to evaluate")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained model")
    parser.add_argument("--gui_aima_code", type=str, default=None,
                        help="Path to GUI-AIMA source code (containing gui_aima/ package)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="JSON path(s), comma-separated")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Image folder(s), comma-separated")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to evaluate (random subset)")
    parser.add_argument("--max_samples_per_dataset", type=str, default=None,
                        help="Per-dataset limits, comma-separated")
    parser.add_argument("--max_pixels", type=int, default=5720064,
                        help="Max pixels for image processing")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Add GUI-AIMA code to path if specified
    if args.gui_aima_code:
        sys.path.insert(0, args.gui_aima_code)
    elif args.model_type in ("gui_aima", "gui_actor"):
        # Try to find gui_aima package from model_path parent
        parent = str(Path(args.model_path).parent)
        if os.path.isdir(os.path.join(parent, "gui_aima")):
            sys.path.insert(0, parent)

    # Load data
    print("Loading datasets...")
    samples = load_datasets(
        args.data_path, args.image_folder,
        max_samples=args.max_samples,
        max_samples_per_dataset=args.max_samples_per_dataset,
        seed=args.seed,
    )

    # Build evaluator
    print(f"Loading {args.model_type} model from {args.model_path}...")
    if args.model_type == "gui_aima":
        evaluator = GUIAIMAEvaluator(args.model_path, args.max_pixels, args.device)
    elif args.model_type == "gui_actor":
        evaluator = GUIActorEvaluator(args.model_path, args.max_pixels, args.device)
    elif args.model_type == "qwen_vanilla":
        evaluator = QwenVanillaEvaluator(args.model_path, args.max_pixels, args.device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Run
    evaluate(evaluator, samples, save_path=args.save_path)


if __name__ == "__main__":
    main()
