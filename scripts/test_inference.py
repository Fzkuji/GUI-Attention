#!/usr/bin/env python3
"""Quick smoke test: load model, run foveated inference on a dummy screenshot."""

import argparse
import time

import torch
from PIL import Image

from gui_attention.model.foveated_qwen25vl import FoveatedQwen25VL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image_path", type=str, default=None, help="Path to test screenshot")
    parser.add_argument("--instruction", type=str, default="Click the search button")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    t0 = time.time()
    model = FoveatedQwen25VL(model_name_or_path=args.model_path)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Create dummy screenshot if no image provided
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
    else:
        print("No image provided, creating dummy 1920x1080 screenshot...")
        image = Image.new("RGB", (1920, 1080), color=(50, 100, 150))

    # Test foveated sampling
    print("\n--- Foveated Sampling ---")
    sampler = model.sampler
    foveated = sampler.sample(image)
    for crop in foveated["crops"]:
        img = crop["image"]
        print(f"  {crop['name']:12s}: {img.size[0]}x{img.size[1]} px, bbox={crop['bbox']}")

    # Test inference
    print(f"\n--- Inference ---")
    print(f"Instruction: {args.instruction}")
    t0 = time.time()
    try:
        x, y = model.predict(image, args.instruction)
        print(f"Predicted click: ({x:.4f}, {y:.4f})")
        print(f"Inference time: {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

    # Token count analysis
    print("\n--- Token Analysis ---")
    from gui_attention.foveation.sampler import FoveatedSampler
    s = FoveatedSampler()
    result = s.sample(image)
    for crop in result["crops"]:
        img = crop["image"]
        # Qwen2.5-VL: ~1 token per 28x28 patch
        est_tokens = (img.size[0] // 28) * (img.size[1] // 28)
        print(f"  {crop['name']:12s}: ~{est_tokens} tokens (est.)")


if __name__ == "__main__":
    main()
