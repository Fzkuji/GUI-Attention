"""
Progressive Resolution Training for GUI Grounding.

Based on GUI-AIMA's attention supervision framework, but with multi-resolution input:
- Image 1 (t=0): Low-resolution full screenshot  
- Image 2 (t=1): High-resolution crop around target region
- Both images share the same ANCHOR token and KL attention supervision

Uses M-RoPE temporal dimension to encode resolution levels.
"""

import copy
import json
import math
import os
import random
import re
import ast
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# Import GUI-AIMA components
# ============================================================
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../Experiments/GUI-AIMA/src"))

from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.trainer import AGUVISTrainer, rank0_print, safe_save_model_for_hf_trainer, EmptyCacheCallback
from gui_aima.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    ADDITIONAL_SPECIAL_TOKENS,
    ACTION_PATTENS_XY,
    chat_template,
    assistant_template,
    grounding_system_message,
)
from gui_aima.dataset import (
    reformat_coordinates,
    get_token_index,
    get_multi_patch_labels,
)


# ============================================================
# Arguments
# ============================================================
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="smz8599/GUI-AIMA-3B")
    weighting: str = field(default="query_1")
    number_of_points: int = field(default=1)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    min_pixels: int = field(default=3136)
    max_pixels: int = field(default=1003520)  # Low-res for full image (~1000 tokens)
    crop_pixels: int = field(default=501760)  # Medium-res for crop (~500 tokens)
    crop_ratio: float = field(default=0.3)    # Crop size relative to full image
    crop_jitter: float = field(default=0.1)   # Random jitter for crop center
    max_conv_turns: int = field(default=10)
    early_mix_text: bool = False


@dataclass 
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=8192)
    gradient_checkpointing: bool = field(default=True)
    pointer_loss_weight: float = field(default=1.0)
    lm_loss_weight: float = field(default=1.0)
    empty_cache_every_n_steps: int = field(default=15)


# ============================================================
# Progressive Dataset
# ============================================================
def compute_crop_bbox(gt_bbox, image_w, image_h, crop_ratio=0.3, jitter=0.1):
    """
    Compute a crop region centered around the GT bbox with jitter.
    Returns (x1, y1, x2, y2) in pixel coordinates.
    """
    # GT center
    gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2 * image_w
    gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2 * image_h
    
    # Crop size
    crop_w = image_w * crop_ratio
    crop_h = image_h * crop_ratio
    
    # Add jitter
    jitter_x = random.uniform(-jitter, jitter) * image_w
    jitter_y = random.uniform(-jitter, jitter) * image_h
    cx = gt_cx + jitter_x
    cy = gt_cy + jitter_y
    
    # Crop bbox
    x1 = max(0, cx - crop_w / 2)
    y1 = max(0, cy - crop_h / 2)
    x2 = min(image_w, cx + crop_w / 2)
    y2 = min(image_h, cy + crop_h / 2)
    
    # Ensure minimum size
    if x2 - x1 < crop_w * 0.5:
        x1 = max(0, x2 - crop_w)
        x2 = min(image_w, x1 + crop_w)
    if y2 - y1 < crop_h * 0.5:
        y1 = max(0, y2 - crop_h)
        y2 = min(image_h, y1 + crop_h)
    
    return int(x1), int(y1), int(x2), int(y2)


def get_multi_image_token_index(processor, images, point_x, point_y, image_idx=0):
    """
    Get visual token index for a point in one of multiple images.
    point_x, point_y are in [0,1] relative to the specified image.
    Returns the token index within that image's visual tokens.
    """
    image = images[image_idx]
    w, h = image.size
    px, py = w * point_x, h * point_y
    
    merge_patch_size = processor.image_processor.patch_size * processor.image_processor.merge_size
    x_index = math.floor(px / merge_patch_size)
    y_index = math.floor(py / merge_patch_size)
    
    visual_token_index = y_index * (w // merge_patch_size) + x_index
    return visual_token_index


def get_multi_image_patch_labels(processor, image, bbox_gt, scheme="gaussian", gaussian_alpha=0.8):
    """
    Get patch-wise labels for a single image (wrapper for compatibility).
    bbox_gt is in [0,1] normalized coordinates relative to this image.
    """
    return get_multi_patch_labels(
        processor.image_processor,
        [image],  # wrap in list for compatibility
        bbox_gt,
        scheme=scheme,
        gaussian_alpha=gaussian_alpha,
    )


class ProgressiveDataset(Dataset):
    """
    Dataset that produces 2-image inputs:
    - Image 1: Low-resolution full screenshot
    - Image 2: High-resolution crop around GT
    
    The conversation format uses two <image> tokens.
    Attention labels are computed for BOTH images' visual tokens.
    """
    
    def __init__(self, tokenizer, processor, data_path, data_args, number_of_points=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor  # This is the main processor (used for full image)
        self.data_args = data_args
        self.number_of_points = number_of_points
        self.model_path = data_args.model_name_or_path_cache
        
        self.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
        self.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
        self.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        
        # Load data
        rank0_print(f"Loading {data_path}")
        with open(data_path) as f:
            self.data = json.load(f)
        rank0_print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        try:
            sample = self._get_item(i)
            if sample is None:
                return self.__getitem__(random.randint(0, len(self.data) - 1))
            return sample
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            return self.__getitem__(random.randint(0, len(self.data) - 1))
    
    def _get_item(self, i):
        item = self.data[i]
        conversations = item["conversations"]
        image_file = item["image"]
        if isinstance(image_file, list):
            image_file = image_file[0]
        
        image_path = os.path.join(self.data_args.image_folder, image_file)
        if not os.path.exists(image_path):
            return None
        
        full_image = Image.open(image_path).convert("RGB")
        img_w, img_h = full_image.size
        
        # Extract GT bbox from conversations
        bbox_gt = None
        for conv in conversations:
            if "bbox_gt" in conv and conv["bbox_gt"] is not None:
                bbox_gt = conv["bbox_gt"]
                break
        
        if bbox_gt is None:
            return None
        
        # Compute crop region around GT
        crop_x1, crop_y1, crop_x2, crop_y2 = compute_crop_bbox(
            bbox_gt, img_w, img_h,
            crop_ratio=self.data_args.crop_ratio,
            crop_jitter=self.data_args.crop_jitter,
        )
        
        # Crop image
        crop_image = full_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_w, crop_h = crop_image.size
        if crop_w < 28 or crop_h < 28:
            return None
        
        # Compute GT coordinates relative to crop
        gt_cx = (bbox_gt[0] + bbox_gt[2]) / 2  # normalized [0,1] in full image
        gt_cy = (bbox_gt[1] + bbox_gt[3]) / 2
        
        # GT in crop coordinates [0,1]
        crop_gt_x = (gt_cx * img_w - crop_x1) / crop_w
        crop_gt_y = (gt_cy * img_h - crop_y1) / crop_h
        
        # Check GT is within crop
        if not (0 < crop_gt_x < 1 and 0 < crop_gt_y < 1):
            return None
        
        # GT bbox in crop coordinates [0,1]
        crop_bbox = [
            max(0, (bbox_gt[0] * img_w - crop_x1) / crop_w),
            max(0, (bbox_gt[1] * img_h - crop_y1) / crop_h),
            min(1, (bbox_gt[2] * img_w - crop_x1) / crop_w),
            min(1, (bbox_gt[3] * img_h - crop_y1) / crop_h),
        ]
        
        # Build conversation with 2 images
        # Format: system + user(image1 + image2 + text) + assistant(action with pointer)
        user_content = conversations[0] if conversations[0].get("from", conversations[0].get("role", "")) in ["human", "user"] else conversations[1]
        asst_content = None
        for conv in conversations:
            role = conv.get("from", conv.get("role", ""))
            if role in ["gpt", "assistant"]:
                asst_content = conv
                break
        
        if asst_content is None:
            return None
        
        user_text = user_content.get("value", user_content.get("content", ""))
        user_text = user_text.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        
        asst_text = asst_content.get("value", asst_content.get("content", ""))
        
        # Reformat coordinates in assistant response
        asst_text_reformatted, coordinates = reformat_coordinates(asst_text, number_of_points=self.number_of_points)
        
        if len(coordinates) == 0:
            return None
        
        # Build messages for tokenization
        # System message
        system_input_ids = self.tokenizer.apply_chat_template(
            conversation=[{"role": "system", "content": [{"type": "text", "text": grounding_system_message}]}],
            chat_template=chat_template,
        )
        
        # User message with 2 images
        user_conv = {
            "role": "user",
            "content": [
                {"type": "image", "image": full_image, "min_pixels": self.data_args.min_pixels, "max_pixels": self.data_args.max_pixels},
                {"type": "image", "image": crop_image, "min_pixels": self.data_args.min_pixels, "max_pixels": self.data_args.crop_pixels},
                {"type": "text", "text": user_text},
            ]
        }
        
        image_inputs, _ = process_vision_info([user_conv])
        
        user_templated = self.tokenizer.apply_chat_template(
            conversation=[user_conv], chat_template=chat_template, tokenize=False
        )
        user_inputs = self.processor(text=[user_templated], images=image_inputs, return_tensors="pt")
        
        pixel_values = user_inputs["pixel_values"]
        image_grid_thw = user_inputs["image_grid_thw"]
        user_input_ids = user_inputs.input_ids[0].tolist()
        
        # Assistant message  
        asst_conv = {
            "role": "assistant",
            "content": [{"type": "text", "text": asst_text_reformatted}],
            "recipient": "os",
            "end_turn": True,
        }
        asst_templated = self.tokenizer.apply_chat_template(
            conversation=[asst_conv], chat_template=assistant_template, tokenize=False
        )
        asst_inputs = self.processor(text=[asst_templated], return_tensors="pt")
        asst_input_ids = asst_inputs.input_ids[0].tolist()
        
        # Combine input_ids
        input_ids = system_input_ids + user_input_ids + asst_input_ids
        labels = [IGNORE_INDEX] * len(system_input_ids) + [IGNORE_INDEX] * len(user_input_ids) + asst_input_ids
        
        # Make pointer_end_token labels IGNORE
        labels = [IGNORE_INDEX if t == self.pointer_end_token_id else t for t in labels]
        
        # Compute visual token indices
        # We need to find which image's visual tokens contain the GT
        # For progressive: we supervise attention on BOTH images
        # Image 1 (full, low-res): GT in full-image coordinates
        # Image 2 (crop, high-res): GT in crop coordinates
        
        # Get resized image sizes from processor
        resized_images = image_inputs  # These are the resized PIL images
        
        # Visual token index in image 1 (full image, low-res)
        visual_token_index_img1 = get_multi_image_token_index(
            self.processor, resized_images, coordinates[0][0], coordinates[0][1], image_idx=0
        )
        
        # Visual token index in image 2 (crop)
        visual_token_index_img2 = get_multi_image_token_index(
            self.processor, resized_images, crop_gt_x, crop_gt_y, image_idx=1
        )
        
        # Compute number of visual tokens per image
        n_vis_tokens_img1 = int(
            image_grid_thw[0][0] * image_grid_thw[0][1] * image_grid_thw[0][2]
            / (self.processor.image_processor.merge_size ** 2)
        )
        
        # Offset for image 2's tokens
        visual_token_index_img2_offset = n_vis_tokens_img1 + visual_token_index_img2
        
        # Use image2's index as the primary target (high-res is more precise)
        visual_token_indices = torch.tensor([visual_token_index_img2_offset], dtype=torch.long)
        
        # Multi-patch labels: combine both images
        # Image 1 labels (low-res, full image)
        patch_labels_img1 = get_multi_image_patch_labels(
            self.processor, resized_images[0], bbox_gt, scheme="gaussian"
        )
        
        # Image 2 labels (high-res, crop)
        patch_labels_img2 = get_multi_image_patch_labels(
            self.processor, resized_images[1], crop_bbox, scheme="gaussian"
        )
        
        # Concatenate: [img1_patches | img2_patches]
        multi_patch_labels = torch.cat([patch_labels_img1, patch_labels_img2], dim=0)
        
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Check length
        n_total_vis = int(sum(
            g[0] * g[1] * g[2] / (self.processor.image_processor.merge_size ** 2)
            for g in image_grid_thw
        ))
        if len(input_ids) + n_total_vis > self.tokenizer.model_max_length:
            return None
        
        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "coordinates": coordinates,
            "visual_token_indices_of_coordinates": visual_token_indices,
            "multi_patch_labels": multi_patch_labels.unsqueeze(0),  # (1, n_vis_total)
        }

    @property
    def lengths(self):
        return [1200] * len(self.data)
    
    @property
    def modality_lengths(self):
        return [1200] * len(self.data)


# ============================================================
# Data Collator
# ============================================================
class ProgressiveDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances):
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        
        # Pad
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        batch = {
            "input_ids": input_ids,
            "labels": labels.long(),
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        
        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.cat([inst["pixel_values"] for inst in instances], dim=0)
            batch["image_grid_thw"] = torch.cat([inst["image_grid_thw"] for inst in instances], dim=0)
        
        if "coordinates" in instances[0]:
            batch["coordinates"] = [inst["coordinates"] for inst in instances]
            batch["visual_token_indices_of_coordinates"] = [
                inst["visual_token_indices_of_coordinates"] for inst in instances
            ]
        
        if "multi_patch_labels" in instances[0]:
            batch["multi_patch_labels"] = [inst["multi_patch_labels"] for inst in instances]
        
        return batch


# ============================================================
# Utility functions (from GUI-AIMA train.py)
# ============================================================
def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = len(tokenizer)
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = new_vocab_size
    else:
        model.config.vocab_size = new_vocab_size
    model.vocab_size = new_vocab_size
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings[-num_new_tokens:] = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)


def update_pointer_token_ids(config, tokenizer, number_of_points=1):
    config.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
    config.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
    config.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]


# ============================================================
# Main Training
# ============================================================
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Store model path for dataset processor creation
    data_args.model_name_or_path_cache = model_args.model_name_or_path
    
    # Load model
    rank0_print(f"Loading model from {model_args.model_name_or_path}...")
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False
    model.reset_loss_weights(
        pointer_loss_weight=training_args.pointer_loss_weight,
        lm_loss_weight=training_args.lm_loss_weight,
    )
    if model_args.weighting:
        model.set_attention_args(weighting=model_args.weighting)
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            model.get_input_embeddings().register_forward_hook(
                lambda module, input, output: output.requires_grad_(True)
            )
    
    # Unfreeze all parameters (full fine-tuning like GUI-AIMA)
    rank0_print("Unfreezing all parameters...")
    for p in model.parameters():
        p.requires_grad = True
    
    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS},
        tokenizer=tokenizer,
        model=model,
    )
    update_pointer_token_ids(model.config, tokenizer)
    
    # Processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
    )
    processor.tokenizer = tokenizer
    
    # Dataset
    train_dataset = ProgressiveDataset(
        tokenizer=tokenizer,
        processor=processor,
        data_path=data_args.data_path,
        data_args=data_args,
        number_of_points=model_args.number_of_points,
    )
    
    data_collator = ProgressiveDataCollator(tokenizer=tokenizer)
    
    # Output dir
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Trainer
    trainer = AGUVISTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[EmptyCacheCallback(every_n_steps=training_args.empty_cache_every_n_steps)],
    )
    
    # New token embedding gradient mask
    emb_param = None
    for n, p in trainer.model.named_parameters():
        if n.endswith("model.embed_tokens.weight"):
            emb_param = p
            break
    if emb_param is not None:
        n_new = len(ADDITIONAL_SPECIAL_TOKENS)
        def mask_grad(grad):
            grad[:-n_new] = 0.0
            return grad
        emb_param.register_hook(mask_grad)
    
    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
