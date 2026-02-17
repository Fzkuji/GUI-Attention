"""Measure GUI-Actor inference time on ScreenSpot-Pro (without modifying their code).

Wraps their inference function with per-sample timing.
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from gui_actor.constants import (
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    chat_template,
)
from gui_actor.inference import ForceFollowTokensLogitsProcessor, inference
from gui_actor.modeling_qwen25vl import (
    Qwen2_5_VLForConditionalGenerationWithPointer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--resize_to_pixels", type=int, default=5760000)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # Load model
    data_processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = data_processor.tokenizer

    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    ).eval()

    logits_processor_pointer = ForceFollowTokensLogitsProcessor(
        token_a_id=tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0],
        forced_sequence=[tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]],
    )

    grounding_system_message = (
        "You are a GUI agent. Given a screenshot of the current GUI and a human "
        "instruction, your task is to locate the screen element that corresponds to "
        "the instruction. You should output a PyAutoGUI action that performs a click "
        "on the correct position. To indicate the click location, we will use some "
        "special tokens, which is used to refer to a visual patch later. For example, "
        "you can output: pyautogui.click(<your_special_token_here>)."
    )

    # Load data
    data_fn = os.path.join(args.data_path, "annotations/all.json")
    with open(data_fn) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    if args.max_samples:
        data = data[: args.max_samples]

    image_dir = os.path.join(args.data_path, "images")
    times = []

    for i, example in tqdm(enumerate(data), total=len(data)):
        image_path = os.path.join(image_dir, example["img_filename"])
        image = Image.open(image_path).convert("RGB")

        if args.resize_to_pixels:
            w, h = image.size
            ratio = (args.resize_to_pixels / (w * h)) ** 0.5
            new_w = round(w * ratio / 28) * 28
            new_h = round(h * ratio / 28) * 28
            image = image.resize((new_w, new_h))

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": example["instruction"]},
                ],
            },
        ]

        t0 = time.time()
        with torch.no_grad():
            pred = inference(
                conversation, model, tokenizer, data_processor,
                logits_processor=logits_processor_pointer,
                use_placeholder=False, topk=3,
            )
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

    # Report
    avg_time = sum(times) / len(times)
    print(f"\n=== GUI-Actor Inference Time ===")
    print(f"Samples: {len(times)}")
    print(f"Avg: {avg_time:.3f}s/sample")
    print(f"Min: {min(times):.3f}s  Max: {max(times):.3f}s")
    print(f"Total: {sum(times):.1f}s")

    # Save
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "timing.json"), "w") as f:
        json.dump({"avg_time": avg_time, "min_time": min(times),
                    "max_time": max(times), "total_time": sum(times),
                    "n_samples": len(times), "per_sample": times}, f, indent=2)
    print(f"Saved timing to {args.save_path}/timing.json")


if __name__ == "__main__":
    main()
