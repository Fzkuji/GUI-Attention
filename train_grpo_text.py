"""
GRPO Training for GUI Grounding with text coordinate output.

Uses Qwen2.5-VL-3B-Instruct directly, no pointer head needed.
Model generates text like "<think>analysis</think><answer>(x, y)</answer>"
Reward = format_reward + gaussian_point_reward based on parsed (x,y) vs GT.
"""

import os
import sys
import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile

import transformers
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
)
from qwen_vl_utils import process_vision_info

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# Arguments
# ============================================================
@dataclass
class ScriptArguments:
    model_name_or_path: str = field(default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    max_samples: Optional[int] = field(default=None)
    min_pixels: int = field(default=3136)
    max_pixels: int = field(default=1003520)
    reward_alpha: float = field(default=0.5)


@dataclass
class GRPOTrainingArguments(transformers.TrainingArguments):
    num_generations: int = field(default=8)
    max_completion_length: int = field(default=256)
    temperature: float = field(default=0.9)
    beta: float = field(default=0.04)
    epsilon: float = field(default=0.2)
    optim: str = field(default="adamw_torch")
    gradient_checkpointing: bool = field(default=True)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)


# ============================================================
# System prompt
# ============================================================
SYSTEM_PROMPT = (
    "You are a GUI grounding assistant. Given a screenshot and an instruction, "
    "locate the target UI element. First analyze the screenshot briefly, "
    "then output the click coordinates.\n"
    "Output format: <think>your analysis</think><answer>x, y</answer>\n"
    "where x and y are normalized coordinates in [0, 1000].\n"
    "Example: <think>The search button is in the top right corner.</think><answer>850, 45</answer>"
)


# ============================================================
# Reward Functions
# ============================================================

def parse_coordinates(text):
    """Parse (x, y) from model output. Returns normalized [0,1] coords or None."""
    # Try <answer>x, y</answer> format first
    match = re.search(r'<answer>\s*(\d+(?:\.\d+)?)\s*[,\s]\s*(\d+(?:\.\d+)?)\s*</answer>', text)
    if match:
        x, y = float(match.group(1)), float(match.group(2))
        # Normalize from [0, 1000] to [0, 1]
        if x > 1:
            x /= 1000
        if y > 1:
            y /= 1000
        return (min(max(x, 0), 1), min(max(y, 0), 1))
    
    # Try pyautogui.click(x=..., y=...) format
    match = re.search(r'click\(x=(\d+(?:\.\d+)?),\s*y=(\d+(?:\.\d+)?)\)', text)
    if match:
        x, y = float(match.group(1)), float(match.group(2))
        if x > 1:
            x /= 1000
        if y > 1:
            y /= 1000
        return (min(max(x, 0), 1), min(max(y, 0), 1))
    
    # Try (x, y) pattern
    match = re.search(r'\((\d+(?:\.\d+)?)\s*[,\s]\s*(\d+(?:\.\d+)?)\)', text)
    if match:
        x, y = float(match.group(1)), float(match.group(2))
        if x > 1:
            x /= 1000
        if y > 1:
            y /= 1000
        return (min(max(x, 0), 1), min(max(y, 0), 1))
    
    return None


def gaussian_point_reward(pred_x, pred_y, gt_bbox, alpha=0.5):
    """Gaussian point reward (GUI-ARP Eq. 2)."""
    gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
    gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
    sigma_x = max(alpha * (gt_bbox[2] - gt_bbox[0]), 0.01)
    sigma_y = max(alpha * (gt_bbox[3] - gt_bbox[1]), 0.01)
    return math.exp(-0.5 * (
        ((pred_x - gt_cx) ** 2) / (sigma_x ** 2) +
        ((pred_y - gt_cy) ** 2) / (sigma_y ** 2)
    ))


def format_reward(text):
    """Check if output has <think>...</think><answer>...</answer> format."""
    has_think = bool(re.search(r'<think>.*?</think>', text, re.DOTALL))
    has_answer = bool(re.search(r'<answer>.*?</answer>', text, re.DOTALL))
    if has_think and has_answer:
        return 1.0
    elif has_answer:
        return 0.5
    return 0.0


def compute_reward(text, gt_bbox, alpha=0.5):
    """Compute total reward for a generation."""
    fmt = format_reward(text)
    
    coord = parse_coordinates(text)
    if coord is None:
        return fmt * 0.5, coord  # Partial format reward, no point reward
    
    pred_x, pred_y = coord
    point = gaussian_point_reward(pred_x, pred_y, gt_bbox, alpha)
    
    return fmt + point, coord


# ============================================================
# Data
# ============================================================

def load_data(data_path, image_folder, max_samples=None):
    with open(data_path) as f:
        raw_data = json.load(f)
    if max_samples:
        random.shuffle(raw_data)
        raw_data = raw_data[:max_samples]
    
    samples = []
    for item in raw_data:
        conversations = item["conversations"]
        image_file = item["image"]
        if isinstance(image_file, list):
            image_file = image_file[0]
        image_path = os.path.join(image_folder, image_file)
        if not os.path.exists(image_path):
            continue
        
        bbox_gt = None
        user_text = ""
        for conv in conversations:
            if "bbox_gt" in conv and conv["bbox_gt"] is not None:
                bbox_gt = conv["bbox_gt"]
            role = conv.get("from", conv.get("role", ""))
            if role in ["human", "user"]:
                text = conv.get("value", conv.get("content", ""))
                user_text = re.sub(r"<image>", "", text).strip()
        
        if bbox_gt is None or not user_text:
            continue
        samples.append({"image_path": image_path, "instruction": user_text, "bbox_gt": bbox_gt})
    
    print(f"Loaded {len(samples)} samples")
    return samples


# ============================================================
# GRPO Trainer
# ============================================================

class TextGRPOTrainer:
    def __init__(self, model, processor, tokenizer, train_data, args, script_args):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.args = args
        self.script_args = script_args
        
        # Reference model for KL
        if args.beta > 0:
            import copy
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        else:
            self.ref_model = None
        
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
        
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        self.metrics = defaultdict(list)
    
    def _build_prompt(self, sample):
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": sample["image_path"]},
                {"type": "text", "text": sample["instruction"]},
            ]}
        ]
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conversation)
        inputs = self.processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
        return inputs
    
    def train_step(self, batch_samples):
        device = self.model.device
        total_loss = 0.0
        n_updates = 0
        
        for sample in batch_samples:
            prompt_inputs = self._build_prompt(sample)
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            # Generate G completions
            completions_text = []
            completions_ids = []
            full_sequences = []
            
            self.model.eval()
            for g in range(self.args.num_generations):
                inputs = {k: v.clone() for k, v in prompt_inputs.items()}
                with torch.no_grad():
                    result = self.model.generate(**inputs, generation_config=self.generation_config)
                comp_ids = result[0, prompt_len:]
                comp_text = self.tokenizer.decode(comp_ids, skip_special_tokens=True)
                completions_text.append(comp_text)
                completions_ids.append(comp_ids)
                full_sequences.append(result[0])
            
            # Compute rewards
            rewards = []
            coords = []
            for text in completions_text:
                r, c = compute_reward(text, sample["bbox_gt"], self.script_args.reward_alpha)
                rewards.append(r)
                coords.append(c)
            
            rewards_tensor = torch.tensor(rewards, device=device)
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std() + 1e-4
            advantages = (rewards_tensor - mean_r) / std_r
            
            # Policy gradient
            self.model.train()
            for comp_ids, full_seq, advantage in zip(completions_ids, full_sequences, advantages):
                if advantage.abs() < 1e-6:
                    continue
                if len(comp_ids) == 0:
                    continue
                
                full_input_ids = full_seq.unsqueeze(0)
                full_attn_mask = torch.ones_like(full_input_ids)
                
                outputs = self.model(
                    input_ids=full_input_ids,
                    attention_mask=full_attn_mask,
                    pixel_values=prompt_inputs.get("pixel_values"),
                    image_grid_thw=prompt_inputs.get("image_grid_thw"),
                )
                
                logits = outputs.logits
                log_probs = F.log_softmax(logits[0, prompt_len-1:-1, :], dim=-1)
                token_ids = full_seq[prompt_len:]
                min_len = min(log_probs.shape[0], len(token_ids))
                per_token_logps = log_probs[:min_len].gather(1, token_ids[:min_len].unsqueeze(-1)).squeeze(-1)
                
                loss = -(advantage * per_token_logps.mean())
                
                # KL penalty
                if self.ref_model is not None and self.args.beta > 0:
                    with torch.no_grad():
                        ref_outputs = self.ref_model(
                            input_ids=full_input_ids,
                            attention_mask=full_attn_mask,
                            pixel_values=prompt_inputs.get("pixel_values"),
                            image_grid_thw=prompt_inputs.get("image_grid_thw"),
                        )
                    ref_log_probs = F.log_softmax(ref_outputs.logits[0, prompt_len-1:-1, :], dim=-1)
                    ref_per_token_logps = ref_log_probs[:min_len].gather(1, token_ids[:min_len].unsqueeze(-1)).squeeze(-1)
                    kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                    loss = loss + self.args.beta * kl.mean()
                
                loss = loss / self.args.num_generations
                loss.backward()
                total_loss += loss.item()
                n_updates += 1
            
            # Log
            self.metrics["reward"].append(mean_r.item())
            self.metrics["reward_std"].append(std_r.item())
            valid_coords = [c for c in coords if c is not None]
            if valid_coords:
                gt_cx = (sample["bbox_gt"][0] + sample["bbox_gt"][2]) / 2
                gt_cy = (sample["bbox_gt"][1] + sample["bbox_gt"][3]) / 2
                avg_dist = sum(
                    math.sqrt((c[0] - gt_cx)**2 + (c[1] - gt_cy)**2) for c in valid_coords
                ) / len(valid_coords)
                self.metrics["avg_dist"].append(avg_dist)
            self.metrics["parse_rate"].append(len(valid_coords) / len(coords))
        
        if n_updates > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return total_loss
    
    def train(self):
        device = self.model.device
        print(f"Starting GRPO training:")
        print(f"  Samples: {len(self.train_data)}")
        print(f"  Epochs: {self.args.num_train_epochs}")
        print(f"  Generations per prompt: {self.args.num_generations}")
        print(f"  Temperature: {self.args.temperature}")
        print(f"  Beta (KL): {self.args.beta}")
        print(f"  Learning rate: {self.args.learning_rate}")
        
        global_step = 0
        for epoch in range(int(self.args.num_train_epochs)):
            random.shuffle(self.train_data)
            for i in range(0, len(self.train_data), self.args.per_device_train_batch_size):
                batch = self.train_data[i:i+self.args.per_device_train_batch_size]
                loss = self.train_step(batch)
                global_step += 1
                
                if global_step % self.args.logging_steps == 0:
                    m = {k: sum(v[-10:])/max(len(v[-10:]),1) for k, v in self.metrics.items()}
                    print(f"[Epoch {epoch+1}] Step {global_step} | "
                          f"Loss: {loss:.4f} | "
                          f"Reward: {m.get('reward',0):.3f} | "
                          f"Dist: {m.get('avg_dist',0):.3f} | "
                          f"Parse: {m.get('parse_rate',0):.1%}")
                
                if global_step % self.args.save_steps == 0:
                    save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    self.model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")
        
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        print(f"Training complete. Model saved to {self.args.output_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = transformers.HfArgumentParser((ScriptArguments, GRPOTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    print(f"Loading model from {script_args.model_name_or_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        script_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )
    model.config.use_cache = False
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    
    for p in model.parameters():
        p.requires_grad = True
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    processor = AutoProcessor.from_pretrained(
        script_args.model_name_or_path,
        min_pixels=script_args.min_pixels,
        max_pixels=script_args.max_pixels,
    )
    
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = load_data(script_args.data_path, script_args.image_folder, script_args.max_samples)
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    trainer = TextGRPOTrainer(
        model=model, processor=processor, tokenizer=tokenizer,
        train_data=train_data, args=training_args, script_args=script_args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
