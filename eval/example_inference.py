"""
GUI-Attention: Example inference using FoveatedQwen25VL.

Usage:
    python eval/example_inference.py \
        --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
        [--image_path /path/to/screenshot.png] \
        [--instruction "Click the search button"] \
        [--grounding_head_path /path/to/grounding_head.pt]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui_attention.model.foveated_qwen25vl import FoveatedQwen25VL


def create_synthetic_screenshot(width=1920, height=1080):
    """Create a synthetic GUI screenshot for testing."""
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 40], fill=(50, 50, 50))
    draw.text((20, 10), "Test Application", fill=(255, 255, 255))
    draw.rectangle([880, 280, 1040, 320], fill=(66, 133, 244))
    draw.text((920, 292), "Search", fill=(255, 255, 255))
    draw.rectangle([100, 100, 500, 140], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.text((110, 112), "Text input field", fill=(150, 150, 150))
    for i, label in enumerate(["Home", "Settings", "Help", "About"]):
        y = 200 + i * 50
        draw.rectangle([50, y, 200, y + 40], fill=(230, 230, 230), outline=(200, 200, 200))
        draw.text((80, y + 10), label, fill=(50, 50, 50))
    return img


def main():
    parser = argparse.ArgumentParser(description="GUI-Attention Example Inference")
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--instruction", type=str, default="Click the search button")
    parser.add_argument("--fixation_x", type=float, default=None)
    parser.add_argument("--fixation_y", type=float, default=None)
    parser.add_argument("--grounding_head_path", type=str, default=None,
                        help="Path to trained grounding head weights (.pt)")
    args = parser.parse_args()

    print("=" * 60)
    print("GUI-Attention: Example Inference")
    print("=" * 60)

    # Load image
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
        print(f"Loaded image: {args.image_path} ({image.size[0]}x{image.size[1]})")
    else:
        image = create_synthetic_screenshot()
        print(f"Created synthetic screenshot: {image.size[0]}x{image.size[1]}")

    # Load model
    print("\nLoading FoveatedQwen25VL...")
    t0 = time.time()
    model = FoveatedQwen25VL(model_name_or_path=args.model_path)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Load trained grounding head if provided
    if args.grounding_head_path:
        state_dict = torch.load(args.grounding_head_path, map_location="cpu")
        model.grounding_head.load_state_dict(state_dict)
        print(f"  Loaded grounding head from {args.grounding_head_path}")

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"  Device: {device}")

    # Trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable} / {total/1e6:.1f}M total")

    # Set fixation
    fixation = None
    if args.fixation_x is not None and args.fixation_y is not None:
        fixation = (args.fixation_x, args.fixation_y)
    print(f"  Fixation: {fixation or 'center (default)'}")

    # Run inference
    print(f"\nInstruction: {args.instruction}")
    t0 = time.time()
    x, y = model.predict(image, args.instruction, fixation=fixation)
    elapsed = time.time() - t0
    print(f"Predicted click: ({x:.4f}, {y:.4f})")
    print(f"Inference time: {elapsed:.2f}s")

    # For synthetic screenshot, the search button is around (0.5, 0.28)
    if not args.image_path:
        gt_x, gt_y = 960 / 1920, 300 / 1080
        dist = ((x - gt_x)**2 + (y - gt_y)**2)**0.5
        print(f"GT (search button): ({gt_x:.4f}, {gt_y:.4f}), distance: {dist:.4f}")

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
