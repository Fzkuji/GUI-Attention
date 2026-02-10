"""
GTA 60K Dataset for GUI-Attention training.

Loads GTA JSON data, applies foveated sampling with GT-based fixation (+noise),
and constructs model inputs for the FoveatedGroundingHead.
"""

import json
import os
import random
import re
from typing import Dict, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import Qwen2_5_VLProcessor

from gui_attention.foveation.sampler import FoveatedSampler


ANCHOR_TOKEN = "<ANCHOR>"


def parse_gta_sample(sample: dict) -> Optional[dict]:
    """Parse a GTA JSON sample to extract instruction, GT coords, and image path.
    
    Returns dict with keys: image, instruction, gt_x, gt_y, bbox_gt (optional)
    or None if parsing fails.
    """
    convs = sample.get("conversations", [])
    if len(convs) < 2:
        return None

    # Human turn: extract instruction (remove <image> tag)
    human_turn = convs[0]
    instruction = human_turn.get("value", "").replace("<image>", "").strip()

    # GPT turn: extract coordinates from pyautogui.click(x=..., y=...)
    gpt_turn = convs[1]
    gpt_value = gpt_turn.get("value", "")
    
    match = re.search(r"x=([\d.]+),\s*y=([\d.]+)", gpt_value)
    if not match:
        return None

    gt_x = float(match.group(1))
    gt_y = float(match.group(2))

    # Clamp to [0, 1]
    gt_x = max(0.0, min(1.0, gt_x))
    gt_y = max(0.0, min(1.0, gt_y))

    result = {
        "image": sample.get("image", ""),
        "instruction": instruction,
        "gt_x": gt_x,
        "gt_y": gt_y,
    }

    bbox_gt = gpt_turn.get("bbox_gt", None)
    if bbox_gt is not None:
        result["bbox_gt"] = bbox_gt

    return result


class GTADataset(Dataset):
    """GTA 60K dataset for training FoveatedGroundingHead.

    Each sample:
    1. Loads image, applies foveated sampling (fixation = GT + noise)
    2. Builds multi-crop input via Qwen2.5-VL processor
    3. Returns input_ids, pixel_values, image_grid_thw, crop metadata, gt_coords

    Args:
        data_path: Path to GTA JSON file.
        images_folder: Root folder for images (prepended to relative paths in JSON).
        processor: Qwen2_5_VLProcessor instance.
        sampler: FoveatedSampler instance.
        fixation_noise_std: Std dev of Gaussian noise added to GT fixation (normalized coords).
        synthetic_mode: If True, generate dummy images when real images not found.
    """

    def __init__(
        self,
        data_path: str,
        images_folder: str,
        processor: Qwen2_5_VLProcessor,
        sampler: FoveatedSampler,
        fixation_noise_std: float = 0.05,
        synthetic_mode: bool = False,
    ):
        super().__init__()
        self.images_folder = images_folder
        self.processor = processor
        self.sampler = sampler
        self.fixation_noise_std = fixation_noise_std
        self.synthetic_mode = synthetic_mode

        # Load and parse data
        with open(data_path) as f:
            raw_data = json.load(f)

        self.samples = []
        for item in raw_data:
            parsed = parse_gta_sample(item)
            if parsed is not None:
                self.samples.append(parsed)

        print(f"GTADataset: loaded {len(self.samples)}/{len(raw_data)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        try:
            return self._get_item(idx)
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def _get_item(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        gt_x, gt_y = sample["gt_x"], sample["gt_y"]
        instruction = sample["instruction"]

        # Load image
        image = self._load_image(sample["image"])

        # Add noise to GT for fixation (simulates imperfect initial fixation)
        fx = gt_x + random.gauss(0, self.fixation_noise_std)
        fy = gt_y + random.gauss(0, self.fixation_noise_std)
        fx = max(0.0, min(1.0, fx))
        fy = max(0.0, min(1.0, fy))

        # Foveated sampling
        foveated = self.sampler.sample(image, fixation=(fx, fy))
        crops = foveated["crops"]

        # Build message with multi-crop images + ANCHOR token
        image_content = []
        for crop in crops:
            image_content.append({"type": "image", "image": crop["image"]})
        image_content.append({
            "type": "text",
            "text": (
                f"The above images show a GUI screenshot at three resolutions: "
                f"fovea (high-res center), parafovea (medium-res), and periphery (full screen low-res). "
                f"Instruction: {instruction}\n{ANCHOR_TOKEN}"
            ),
        })
        messages = [{"role": "user", "content": image_content}]

        # Process through Qwen2.5-VL processor
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        crop_images = [c["image"] for c in crops]
        inputs = self.processor(
            text=[text],
            images=crop_images,
            padding=False,
            return_tensors="pt",
        )

        # Squeeze batch dim
        input_ids = inputs["input_ids"].squeeze(0)
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]

        # Store crop bboxes and levels for coordinate computation
        crop_bboxes = [c["bbox"] for c in crops]
        crop_levels = [c["level"] for c in crops]

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "gt_coords": torch.tensor([gt_x, gt_y], dtype=torch.float32),
            "crop_bboxes": crop_bboxes,  # list of tuples
            "crop_levels": crop_levels,  # list of ints
        }

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from path, with synthetic fallback."""
        full_path = os.path.join(self.images_folder, image_path)
        if os.path.exists(full_path):
            return Image.open(full_path).convert("RGB")

        if self.synthetic_mode:
            return self._create_synthetic_image()

        raise FileNotFoundError(f"Image not found: {full_path}")

    @staticmethod
    def _create_synthetic_image(width=1920, height=1080) -> Image.Image:
        """Create a dummy screenshot for testing."""
        from PIL import ImageDraw
        img = Image.new("RGB", (width, height), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        # Random UI elements
        for _ in range(random.randint(5, 15)):
            x1 = random.randint(0, width - 200)
            y1 = random.randint(0, height - 50)
            x2 = x1 + random.randint(50, 200)
            y2 = y1 + random.randint(20, 50)
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            draw.rectangle([x1, y1, x2, y2], fill=color)
        return img


class GTACollator:
    """Collator for GTADataset that pads input_ids and stacks tensors."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids_list = [item["input_ids"] for item in batch]
        max_len = max(ids.shape[0] for ids in input_ids_list)

        # Pad input_ids
        padded_ids = []
        attention_masks = []
        for ids in input_ids_list:
            pad_len = max_len - ids.shape[0]
            padded_ids.append(torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)]))
            attention_masks.append(torch.cat([torch.ones(ids.shape[0], dtype=torch.long),
                                               torch.zeros(pad_len, dtype=torch.long)]))

        result = {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(attention_masks),
            "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
            "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch], dim=0),
            "gt_coords": torch.stack([item["gt_coords"] for item in batch]),
            "crop_bboxes": [item["crop_bboxes"] for item in batch],
            "crop_levels": [item["crop_levels"] for item in batch],
        }
        return result
