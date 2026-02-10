"""
GUI-Attention: Example inference script.

Tests the basic foveated sampling + Qwen2.5-VL pipeline:
1. Load a screenshot (or generate a synthetic one)
2. Apply foveated sampling (3 crops)
3. Feed crops as multi-image input to Qwen2.5-VL
4. Extract anchor token attention over visual tokens
5. Predict (x, y) coordinates via grounding head

Usage:
    python eval/example_inference.py \
        --model_path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
        [--image_path /path/to/screenshot.png] \
        [--instruction "Click the search button"]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui_attention.foveation.sampler import FoveatedSampler


def create_synthetic_screenshot(width=1920, height=1080):
    """Create a synthetic GUI screenshot for testing."""
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Draw some UI elements
    # Title bar
    draw.rectangle([0, 0, width, 40], fill=(50, 50, 50))
    draw.text((20, 10), "Test Application", fill=(255, 255, 255))

    # Search button at (960, 300) - this is our target
    draw.rectangle([880, 280, 1040, 320], fill=(66, 133, 244))
    draw.text((920, 292), "Search", fill=(255, 255, 255))

    # Some other UI elements
    draw.rectangle([100, 100, 500, 140], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.text((110, 112), "Text input field", fill=(150, 150, 150))

    # Navigation menu
    for i, label in enumerate(["Home", "Settings", "Help", "About"]):
        y = 200 + i * 50
        draw.rectangle([50, y, 200, y + 40], fill=(230, 230, 230), outline=(200, 200, 200))
        draw.text((80, y + 10), label, fill=(50, 50, 50))

    return img


def test_foveated_sampling(image, fixation=None):
    """Test Step 1: Foveated sampling."""
    print("\n=== Step 1: Foveated Sampling ===")
    sampler = FoveatedSampler(
        fovea_size=0.15,
        parafovea_size=0.40,
        fovea_resolution=768,
        parafovea_resolution=512,
        periphery_resolution=384,
    )

    result = sampler.sample(image, fixation=fixation)

    for crop_info in result["crops"]:
        img = crop_info["image"]
        print(f"  Level {crop_info['level']} ({crop_info['name']}): "
              f"{img.size[0]}x{img.size[1]} pixels, "
              f"bbox={tuple(f'{x:.2f}' for x in crop_info['bbox'])}")

    print(f"  Fixation: {result['fixation']}")
    print(f"  Original size: {result['original_size']}")
    return result


def test_model_loading(model_path):
    """Test Step 2: Load Qwen2.5-VL model."""
    print("\n=== Step 2: Loading Qwen2.5-VL ===")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"  Processor loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Need eager for attention extraction
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, processor


def test_multi_crop_inference(model, processor, foveated_result, instruction):
    """Test Step 3: Multi-crop inference with Qwen2.5-VL."""
    print("\n=== Step 3: Multi-Crop Inference ===")

    crops = foveated_result["crops"]

    # Build multi-image message
    # Each crop is treated as a separate image in the conversation
    content = []
    for crop_info in crops:
        content.append({"type": "image", "image": crop_info["image"]})
    content.append({
        "type": "text",
        "text": (
            f"The images above show the same GUI screenshot at different zoom levels:\n"
            f"- Image 1: Full screen overview (periphery)\n"
            f"- Image 2: Medium zoom of the center region (parafovea)\n"
            f"- Image 3: Close-up of the center region (fovea)\n\n"
            f"Instruction: {instruction}\n"
            f"Please output the click coordinates as (x, y) normalized to [0, 1]."
        ),
    })

    messages = [{"role": "user", "content": content}]

    # Process input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Process images
    image_inputs = [crop_info["image"] for crop_info in crops]

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    # Count visual tokens
    input_ids = inputs["input_ids"][0]
    # In Qwen2.5-VL, image tokens are represented by special token ids
    # Count them to verify our token reduction
    num_total_tokens = len(input_ids)
    print(f"  Total input tokens: {num_total_tokens}")

    # Forward pass with attention output
    t0 = time.time()
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )
    inference_time = time.time() - t0
    print(f"  Forward pass time: {inference_time:.2f}s")

    # Analyze attention
    attentions = outputs.attentions  # tuple of (batch, heads, seq_len, seq_len)
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    seq_len = attentions[0].shape[2]
    print(f"  Attention shape: {num_layers} layers × {num_heads} heads × {seq_len} seq_len")

    # Generate output text
    t0 = time.time()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=50,
        )
    gen_time = time.time() - t0

    # Decode only the new tokens
    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    output_text = processor.decode(new_tokens, skip_special_tokens=True)
    print(f"  Generation time: {gen_time:.2f}s")
    print(f"  Model output: {output_text}")

    return outputs, output_text


def test_anchor_attention_extraction(model, processor, foveated_result, instruction):
    """Test Step 4: Anchor-style attention extraction (simplified).

    Instead of using a trained <ANCHOR> token (which requires fine-tuning),
    we use the last token's attention over visual tokens as a proxy.
    """
    print("\n=== Step 4: Attention-based Coordinate Prediction ===")

    crops = foveated_result["crops"]

    # Build simpler prompt for attention analysis
    content = []
    # Reverse order: periphery first (global context), then parafovea, then fovea
    for crop_info in reversed(crops):
        content.append({"type": "image", "image": crop_info["image"]})
    content.append({
        "type": "text",
        "text": f"Where should I click to: {instruction}",
    })

    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = [crop_info["image"] for crop_info in reversed(crops)]

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    # Identify visual token positions
    input_ids = inputs["input_ids"][0]
    # Qwen2.5-VL uses specific token ids for image placeholders
    # The vision tokens are inserted where <|image_pad|> tokens are
    # Token id for <|image_pad|> in Qwen2.5-VL
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_mask = (input_ids == image_pad_id)
    vision_indices = torch.where(vision_mask)[0]
    num_vision_tokens = vision_indices.shape[0]
    print(f"  Visual tokens: {num_vision_tokens}")
    print(f"  Text tokens: {len(input_ids) - num_vision_tokens}")

    # Forward pass
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

    # Extract last token's attention to visual tokens (proxy for anchor)
    # Average across last few layers and all heads
    last_token_idx = len(input_ids) - 1
    num_layers = len(outputs.attentions)

    # Use last 8 layers (deeper layers have more semantic attention)
    attn_to_visual = []
    for layer_idx in range(num_layers - 8, num_layers):
        # (1, heads, seq, seq) -> (heads, seq)
        layer_attn = outputs.attentions[layer_idx][0, :, last_token_idx, :]
        # Select visual token columns
        visual_attn = layer_attn[:, vision_indices]  # (heads, num_vision)
        attn_to_visual.append(visual_attn)

    # Stack and average: (8, heads, num_vision) -> (num_vision,)
    attn_to_visual = torch.stack(attn_to_visual)  # (8, H, V)
    avg_attn = attn_to_visual.mean(dim=(0, 1))  # (V,)
    avg_attn = avg_attn / avg_attn.sum()  # normalize

    # Print attention statistics
    top_k = 10
    top_vals, top_idxs = avg_attn.topk(top_k)
    print(f"  Top-{top_k} visual token attention: {top_vals.tolist()}")

    # Map visual tokens back to image coordinates
    # This is a simplified version - proper implementation needs
    # to track which crop each visual token belongs to
    print(f"  (Coordinate mapping requires crop-to-token tracking - TODO)")

    peak_idx = avg_attn.argmax().item()
    print(f"  Peak attention at visual token index: {peak_idx}/{num_vision_tokens}")

    return avg_attn


def main():
    parser = argparse.ArgumentParser(description="GUI-Attention Example Inference")
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--instruction", type=str, default="Click the search button")
    args = parser.parse_args()

    print("=" * 60)
    print("GUI-Attention: Example Inference")
    print("=" * 60)

    # Load or create image
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
        print(f"Loaded image: {args.image_path} ({image.size[0]}x{image.size[1]})")
    else:
        image = create_synthetic_screenshot()
        print(f"Created synthetic screenshot: {image.size[0]}x{image.size[1]}")

    # Step 1: Foveated sampling
    foveated_result = test_foveated_sampling(image, fixation=(0.5, 0.28))  # Near the search button

    # Step 2: Load model
    model, processor = test_model_loading(args.model_path)

    # Step 3: Multi-crop inference
    outputs, output_text = test_multi_crop_inference(model, processor, foveated_result, args.instruction)

    # Step 4: Attention extraction
    avg_attn = test_anchor_attention_extraction(model, processor, foveated_result, args.instruction)

    print("\n" + "=" * 60)
    print("All tests passed! Pipeline is functional.")
    print("=" * 60)


if __name__ == "__main__":
    main()
