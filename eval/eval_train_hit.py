"""
Evaluate GUI-Actor's single-round hit rate on the training dataset.

Loads the original GUI-Actor model (Qwen2_5_VLForConditionalGeneration + pointer head)
and runs single-round forward pass to compute hit rate.

Usage:
  cd /mnt/data/zichuanfu/GUI-Attention-Workspace/GUI-Attention
  PYTHONPATH=src:$PYTHONPATH HF_HUB_OFFLINE=1 python eval/eval_train_hit.py \
      --model_path /mnt/data/zichuanfu/GUI-Attention-Workspace/models/GUI-Actor-3B-Qwen2.5-VL \
      --data_path /mnt/data/zichuanfu/GUI-Attention-Workspace/data/guiact_bbox.json,/mnt/data/zichuanfu/GUI-Attention-Workspace/data/androidcontrol_bbox.json \
      --image_folder /mnt/data/zichuanfu/GUI-Attention-Workspace/data,/mnt/data/zichuanfu/GUI-Attention-Workspace/data \
      --max_samples 2000
"""

import argparse
import json
import os
import random
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True

from gui_attention.constants import (
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    GROUNDING_SYSTEM_MESSAGE,
)
from gui_attention.dual_head import _AttentionHead


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
    paths = [p.strip() for p in data_paths.split(",")]
    folders = [f.strip() for f in image_folders.split(",")] if image_folders else [""] * len(paths)

    if len(folders) < len(paths):
        folders.extend([""] * (len(paths) - len(folders)))

    for dp, img_dir in zip(paths, folders):
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

            bbox = item.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue

            instruction = ""
            convs = item.get("conversations", [])
            for c in convs:
                if c.get("from") == "human":
                    instruction = c.get("value", "")
                    instruction = instruction.replace("<image>\n", "").replace("<image>", "").strip()
                    break

            samples.append({
                "image_path": img_path,
                "instruction": instruction,
                "bbox": bbox,
            })
            count += 1

        print(f"  ✓ {os.path.basename(dp)}: {count} samples")

    total = len(samples)
    if max_samples and total > max_samples:
        random.seed(42)
        samples = random.sample(samples, max_samples)
        print(f"  Sampled {max_samples} from {total} total")

    print(f"Total: {len(samples)} samples")
    return samples


# ---------------------------------------------------------------------------
# Model loading: GUI-Actor original (backbone + pointer head)
# ---------------------------------------------------------------------------

def load_gui_actor_model(model_path, device="cuda:0"):
    """Load GUI-Actor model: Qwen2.5-VL backbone + pointer head from safetensors."""
    print(f"Loading backbone from {model_path}...")
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    # Load processor/tokenizer
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    # Build pointer head with the same architecture expected by the saved weights.
    d_model = backbone.config.hidden_size
    pointer_head = _AttentionHead(d_model=d_model, projection_dim=d_model)

    # Load pointer head weights from safetensors
    import glob
    from safetensors.torch import load_file
    sf_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    loaded = 0
    for sf in sf_files:
        state_dict = load_file(sf)
        # GUI-Actor pointer head keys: multi_patch_pointer_head.*
        pointer_state = {}
        for k, v in state_dict.items():
            if k.startswith("multi_patch_pointer_head."):
                new_k = k.replace("multi_patch_pointer_head.", "")
                # Map GUI-Actor names to our _AttentionHead names.
                new_k = new_k.replace("projection_enc", "mlp_v")
                new_k = new_k.replace("projection_dec", "mlp_t")
                pointer_state[new_k] = v
        if pointer_state:
            missing, unexpected = pointer_head.load_state_dict(pointer_state, strict=False)
            loaded += len(pointer_state)
            if missing:
                print(f"  WARNING: missing pointer keys: {missing}")
    print(f"  Pointer head loaded: {loaded} params")

    pointer_head = pointer_head.to(dtype=torch.bfloat16, device=device).eval()

    # Get pointer pad token id
    pp_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)

    return backbone, pointer_head, processor, tokenizer, pp_id


# ---------------------------------------------------------------------------
# Single-round evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_single_round(backbone, pointer_head, processor, tokenizer, pp_id,
                      samples, device="cuda:0"):
    """Run single-round forward pass (GUI-Actor style), compute hit rate."""
    img_tok_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")

    hits = 0
    total = 0
    errors = 0

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            errors += 1
            continue

        img_w, img_h = img.size
        bbox = normalize_bbox(sample["bbox"], img_w, img_h)

        # Build conversation (GUI-Actor format)
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": GROUNDING_SYSTEM_MESSAGE}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": sample["instruction"]},
            ]},
        ]

        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        text += f"pyautogui.click({DEFAULT_POINTER_START_TOKEN}{DEFAULT_POINTER_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN})"

        inputs = processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Forward pass
        outputs = backbone(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer

        input_ids = inputs["input_ids"][0]

        # Find pointer_pad position
        pad_positions = (input_ids == pp_id).nonzero(as_tuple=True)[0]
        if len(pad_positions) == 0:
            errors += 1
            continue
        anchor_hs = hidden_states[0, pad_positions[-1]:pad_positions[-1]+1, :]  # [1, D]

        # Find visual token range
        vis_mask = (input_ids == img_tok_id)
        vis_indices = vis_mask.nonzero(as_tuple=True)[0]
        if len(vis_indices) == 0:
            errors += 1
            continue
        vis_start = vis_indices[0].item()
        vis_end = vis_indices[-1].item() + 1
        n_vis = vis_end - vis_start

        # Get visual encoder embeddings for pointer head
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # Extract ViT embeddings
        _inner = backbone
        if hasattr(_inner, 'model'):
            _inner = _inner.model
        vis_out = _inner.visual(pixel_values.to(_inner.visual.dtype), grid_thw=image_grid_thw)
        vis_embeds = vis_out  # [N_vis, D]

        # Pointer head forward
        attn_weights, _, _ = pointer_head(
            vis_embeds,   # [N_vis, D]
            anchor_hs,    # [1, D]
        )

        # Get predicted position (argmax)
        pred_idx = attn_weights.squeeze(0).argmax().item()

        # Compute grid dimensions from image_grid_thw
        t, h, w = image_grid_thw[0].tolist()
        merge = getattr(_inner.visual if hasattr(_inner, 'visual') else backbone.visual,
                        'spatial_merge_size', 2)
        grid_h = int(h // merge)
        grid_w = int(w // merge)
        n_expected = grid_h * grid_w
        if n_expected != n_vis:
            # Multiple temporal frames or mismatch, use aspect ratio fallback
            aspect = img_w / img_h
            grid_h = max(1, int((n_vis / aspect) ** 0.5))
            grid_w = max(1, n_vis // grid_h)
            while grid_h * grid_w < n_vis:
                grid_w += 1

        row = pred_idx // grid_w
        col = pred_idx % grid_w

        pred_x = (col + 0.5) / grid_w
        pred_y = (row + 0.5) / grid_h

        if point_in_bbox(pred_x, pred_y, bbox):
            hits += 1

        total += 1

        if total % 200 == 0:
            print(f"  Progress: {total} samples, hit={hits}/{total} ({hits/max(total,1)*100:.1f}%)")

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
    parser.add_argument("--data_path", type=str, required=True, help="Comma-separated JSON data paths")
    parser.add_argument("--image_folder", type=str, default="", help="Comma-separated image folders")
    parser.add_argument("--max_samples", type=int, default=2000, help="Max samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    samples = load_datasets(args.data_path, args.image_folder, args.max_samples)

    print(f"Loading GUI-Actor model from {args.model_path}...")
    backbone, pointer_head, processor, tokenizer, pp_id = load_gui_actor_model(
        args.model_path, device=args.device
    )

    print(f"Evaluating on {len(samples)} samples...")
    eval_single_round(backbone, pointer_head, processor, tokenizer, pp_id,
                      samples, device=args.device)
