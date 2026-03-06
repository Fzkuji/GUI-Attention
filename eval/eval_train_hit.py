"""
Evaluate a model's single-round hit rate on the training dataset.

Works with both:
  - GUI-Actor original model (Qwen2_5_VLForConditionalGenerationWithPointer)
  - Our DualHead model (Qwen25VLWithDualHead, single-round mode)

Usage (GUI-Actor baseline):
  PYTHONPATH=src:$PYTHONPATH python eval/eval_train_hit.py \
      --model_path /path/to/GUI-Actor-3B-Qwen2.5-VL \
      --data_path /path/to/guiact_bbox.json,/path/to/androidcontrol_bbox.json \
      --image_folder /path/to/images \
      --max_samples 2000

Usage (our checkpoint):
  PYTHONPATH=src:$PYTHONPATH python eval/eval_train_hit.py \
      --model_path /path/to/GUI-Actor-3B-Qwen2.5-VL \
      --checkpoint /path/to/checkpoint-500 \
      --data_path /path/to/guiact_bbox.json \
      --image_folder /path/to/images \
      --max_samples 2000
"""

import argparse
import json
import os
import random

import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def point_in_bbox(px, py, bbox):
    """Check if (px, py) is inside bbox [x1, y1, x2, y2], all in [0,1]."""
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def normalize_bbox(bbox, img_w, img_h):
    """Normalize bbox to [0,1] if not already."""
    x1, y1, x2, y2 = bbox
    if all(0 <= v <= 1 for v in [x1, y1, x2, y2]):
        return bbox
    return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]


# ---------------------------------------------------------------------------
# Data loading (same format as train.py)
# ---------------------------------------------------------------------------

def load_datasets(data_paths, image_folders, max_samples=None):
    """Load training data from JSON files."""
    samples = []
    paths = data_paths.split(",")
    folders = image_folders.split(",") if image_folders else [""] * len(paths)
    
    for dp, img_dir in zip(paths, folders):
        dp = dp.strip()
        img_dir = img_dir.strip()
        if not os.path.exists(dp):
            print(f"  ✗ {dp} not found, skipping")
            continue
        with open(dp) as f:
            raw = json.load(f)
        
        count = 0
        for item in raw:
            img_path = item.get("image", "")
            if img_dir and not os.path.isabs(img_path):
                img_path = os.path.join(img_dir, img_path)
            
            # Get bbox
            bbox = item.get("bbox")
            if bbox is None:
                continue
            
            # Normalize bbox format: ensure [x1, y1, x2, y2]
            if len(bbox) == 4:
                # Could be [x1, y1, x2, y2] or [x, y, w, h]
                # Training data uses [x1, y1, x2, y2] in normalized coords
                pass
            else:
                continue
            
            instruction = ""
            convs = item.get("conversations", [])
            for c in convs:
                if c.get("from") == "human":
                    instruction = c.get("value", "")
                    # Remove <image> tag
                    instruction = instruction.replace("<image>\n", "").replace("<image>", "").strip()
                    break
            
            samples.append({
                "image_path": img_path,
                "instruction": instruction,
                "bbox": bbox,
            })
            count += 1
        
        print(f"  ✓ {os.path.basename(dp)}: {count} samples")
    
    if max_samples and len(samples) > max_samples:
        random.seed(42)
        samples = random.sample(samples, max_samples)
        print(f"  Sampled {max_samples} from {len(samples)} total")
    
    print(f"Total: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_gui_actor_model(model_path):
    """Load original GUI-Actor model."""
    from gui_attention.model import Qwen2_5_VLForConditionalGenerationWithPointer
    
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def load_dual_head_model(model_path, checkpoint_path):
    """Load our DualHead model from checkpoint."""
    from gui_attention.model import build_model
    
    model, processor = build_model(
        model_name_or_path=model_path,
        use_lora=True,
        lora_r=32,
        lora_alpha=64,
        lora_target_modules="q_proj,v_proj",
        use_dual_tokens=True,
        click_head_from=model_path,
    )
    
    # Load checkpoint weights
    import glob
    safetensor_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file
        for sf in safetensor_files:
            state_dict = load_file(sf)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded {sf}: {len(state_dict)} params, missing={len(missing)}, unexpected={len(unexpected)}")
    
    model = model.to("cuda:0").eval()
    return model, processor


# ---------------------------------------------------------------------------
# Single-round inference (GUI-Actor style)
# ---------------------------------------------------------------------------

GROUNDING_SYSTEM_MESSAGE = (
    "You are a GUI agent. Given a screenshot of the current GUI and a human instruction, "
    "your task is to locate the screen element that corresponds to the instruction. "
    "You should output a PyAutoGUI action that performs a click on the correct position. "
    "To indicate the click location, we will use some special tokens, which is used to refer "
    "to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>)."
)


@torch.no_grad()
def eval_single_round(model, processor, samples, max_pixels=1001600, is_dual_head=False):
    """Run single-round forward pass, compute hit rate."""
    tokenizer = processor.tokenizer
    
    # Get pointer_pad token id
    pointer_pad_id = tokenizer.convert_tokens_to_ids("<|pointer_pad|>")
    
    hits = 0
    total = 0
    errors = 0
    
    for sample in tqdm(samples, desc="Evaluating"):
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            errors += 1
            continue
        
        img_w, img_h = img.size
        bbox = normalize_bbox(sample["bbox"], img_w, img_h)
        
        # Build conversation
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": GROUNDING_SYSTEM_MESSAGE}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": sample["instruction"]},
            ]},
        ]
        
        # Process inputs
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        # Add pointer suffix
        text += "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
        
        inputs = processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Forward pass
        if is_dual_head:
            # For DualHead, we need to use the backbone directly
            outputs = model.backbone(**inputs, output_hidden_states=True)
        else:
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[-1]  # Last layer
        
        # Find pointer_pad position
        input_ids = inputs["input_ids"][0]
        pad_positions = (input_ids == pointer_pad_id).nonzero(as_tuple=True)[0]
        if len(pad_positions) == 0:
            errors += 1
            continue
        pad_pos = pad_positions[-1].item()
        
        # Get anchor hidden state (at pointer_pad position)
        anchor_hs = hidden_states[0, pad_pos:pad_pos+1, :]  # [1, D]
        
        # Get visual token hidden states
        # Find image token range (vision_start to vision_end)
        vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        
        vs_positions = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
        ve_positions = (input_ids == vision_end_id).nonzero(as_tuple=True)[0]
        
        if len(vs_positions) == 0 or len(ve_positions) == 0:
            errors += 1
            continue
        
        # First image tokens (round 0)
        vis_start = vs_positions[0].item() + 1  # After vision_start
        vis_end = ve_positions[0].item()  # Before vision_end
        
        visual_hs = hidden_states[0, vis_start:vis_end, :]  # [N_vis, D]
        
        # Compute attention: pointer head style
        if is_dual_head:
            head = model.click_head
        else:
            head = model.multi_patch_pointer_head
        
        # Forward through pointer head
        attn_weights, loss, logits = head(
            visual_hs.unsqueeze(0),  # [1, N_vis, D]
            anchor_hs.unsqueeze(0),  # [1, 1, D]
        )[:3]
        
        # Get predicted position (argmax)
        pred_idx = attn_weights[0].argmax().item()
        n_vis = vis_end - vis_start
        
        # Convert to grid coordinates
        # Need to figure out the grid dimensions from image size
        from qwen_vl_utils import process_vision_info
        
        # Use the model's image processing to get grid dims
        # Simplified: compute from token count
        # Qwen2.5-VL: merge_patch_size=2, each token = 28x28 pixels
        # After smart_resize, the grid is (H/28) x (W/28)
        # But we don't know exact resize. Use token count + aspect ratio.
        
        aspect = img_w / img_h
        grid_h = int((n_vis / aspect) ** 0.5)
        grid_w = n_vis // grid_h if grid_h > 0 else n_vis
        # Adjust
        while grid_h * grid_w < n_vis and grid_h > 0:
            grid_w += 1
        while grid_h * grid_w > n_vis and grid_w > 0:
            grid_h = n_vis // grid_w
        
        if grid_h * grid_w != n_vis:
            # Fallback: linear
            grid_h = 1
            grid_w = n_vis
        
        row = pred_idx // grid_w
        col = pred_idx % grid_w
        
        pred_x = (col + 0.5) / grid_w
        pred_y = (row + 0.5) / grid_h
        
        if point_in_bbox(pred_x, pred_y, bbox):
            hits += 1
        
        total += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {hits}/{total} = {hits/max(total,1)*100:.2f}% hit")
    print(f"Errors (skipped): {errors}")
    print(f"{'='*50}")
    return hits, total, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate single-round hit rate on training data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to GUI-Actor model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to our DualHead checkpoint (if evaluating our model)")
    parser.add_argument("--data_path", type=str, required=True, help="Comma-separated JSON data paths")
    parser.add_argument("--image_folder", type=str, default="", help="Comma-separated image folders")
    parser.add_argument("--max_samples", type=int, default=2000, help="Max samples to evaluate")
    parser.add_argument("--max_pixels", type=int, default=1001600, help="Max pixels for image resize")
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_path}...")
    samples = load_datasets(args.data_path, args.image_folder, args.max_samples)
    
    is_dual_head = args.checkpoint is not None
    
    if is_dual_head:
        print(f"Loading DualHead model from {args.model_path} + {args.checkpoint}...")
        model, processor = load_dual_head_model(args.model_path, args.checkpoint)
    else:
        print(f"Loading GUI-Actor model from {args.model_path}...")
        model, processor = load_gui_actor_model(args.model_path)
    
    print(f"Evaluating {'DualHead' if is_dual_head else 'GUI-Actor'} on {len(samples)} samples...")
    eval_single_round(model, processor, samples, max_pixels=args.max_pixels, is_dual_head=is_dual_head)
