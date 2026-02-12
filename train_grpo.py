"""
GRPO Training for Iterative Attention-Guided Foveation.

Based on VLM-R1 GRPO framework + GUI-AIMA attention-based grounding.

Flow:
1. Model generates: "分析文本... pyautogui.click(<pointer_pad>)"
2. From <pointer_pad>'s attention over visual patches → extract coordinate
3. Gaussian point reward: predicted coord vs GT bbox
4. GRPO updates text generation policy (which indirectly affects attention)

Usage:
    python train_grpo.py \\
        --model_name_or_path /root/autodl-tmp/models/GUI-AIMA-3B \\
        --data_path /root/autodl-tmp/data/GUI-Actor/guiact_bbox.json \\
        --image_folder /root/autodl-tmp/data/GUI-Actor/images/GUIAct/web_imgs \\
        --output_dir /root/autodl-tmp/checkpoints/grpo_foveation \\
        ...
"""

import os
import sys
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union, Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from datasets import Dataset as HFDataset

import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# GUI-AIMA imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../Experiments/GUI-AIMA/src"))

from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.constants import (
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    ADDITIONAL_SPECIAL_TOKENS,
    chat_template,
    grounding_system_message,
)
from gui_aima.inference import (
    calculate_attention_from_qk,
    get_prediction_region_point,
)

from qwen_vl_utils import process_vision_info


# ============================================================
# Arguments
# ============================================================
@dataclass
class GRPOScriptArguments:
    model_name_or_path: str = field(default="/root/autodl-tmp/models/GUI-AIMA-3B")
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    max_samples: Optional[int] = field(default=None)
    min_pixels: int = field(default=3136)
    max_pixels: int = field(default=1003520)
    reward_alpha: float = field(default=0.5, metadata={"help": "Gaussian reward sigma scaling"})


@dataclass
class GRPOTrainingArguments(transformers.TrainingArguments):
    # GRPO specific
    num_generations: int = field(default=8, metadata={"help": "Number of generations per prompt (G)"})
    max_completion_length: int = field(default=256, metadata={"help": "Max tokens to generate"})
    temperature: float = field(default=0.9, metadata={"help": "Sampling temperature"})
    beta: float = field(default=0.04, metadata={"help": "KL penalty coefficient"})
    epsilon: float = field(default=0.2, metadata={"help": "PPO clip range"})
    num_iterations: int = field(default=1, metadata={"help": "Number of policy updates per batch (mu)"})
    
    # Training
    optim: str = field(default="adamw_torch")
    gradient_checkpointing: bool = field(default=True)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)


# ============================================================
# Reward Functions
# ============================================================

def gaussian_point_reward(pred_x, pred_y, gt_bbox, alpha=0.5):
    """
    Gaussian point reward from GUI-ARP (Eq. 2).
    Continuous reward in [0, 1] based on distance to GT center,
    with sigma proportional to element size.
    """
    gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
    gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
    sigma_x = max(alpha * (gt_bbox[2] - gt_bbox[0]), 0.01)
    sigma_y = max(alpha * (gt_bbox[3] - gt_bbox[1]), 0.01)
    
    return math.exp(-0.5 * (
        ((pred_x - gt_cx) ** 2) / (sigma_x ** 2) +
        ((pred_y - gt_cy) ** 2) / (sigma_y ** 2)
    ))


def extract_attention_coordinate(model, prompt_input_ids, prompt_inputs, image_grid_thw):
    """
    Extract predicted coordinate from <pointer_pad> token's attention map.
    
    Does a fresh forward pass (prefill only, no generation) on the full
    prompt+completion sequence to get all hidden states, then uses 
    GUI-AIMA's QK-recompute attention + pointer head.
    
    Args:
        model: GUI-AIMA model
        prompt_input_ids: Full sequence (prompt + completion) input_ids, shape (seq_len,)
        prompt_inputs: Dict with pixel_values, image_grid_thw, attention_mask for forward
        image_grid_thw: Image grid info for coordinate conversion
    
    Returns:
        (pred_x, pred_y) in [0, 1] or None
    """
    device = prompt_input_ids.device
    input_ids = prompt_input_ids
    
    # Find visual token indices
    visual_mask = (input_ids == model.config.image_token_id)
    visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1)
    if len(visual_indices) == 0:
        return None
    
    # Find pointer_pad token (ANCHOR)
    pointer_pad_ids = model.config.pointer_pad_token_id
    if isinstance(pointer_pad_ids, int):
        pointer_pad_ids = [pointer_pad_ids]
    pointer_pad_mask = torch.isin(input_ids, torch.tensor(pointer_pad_ids, device=device))
    target_indices = torch.nonzero(pointer_pad_mask, as_tuple=False).squeeze(-1)
    if len(target_indices) == 0:
        return None
    
    # Query indices: text tokens between last visual token and first pointer token
    query_start = visual_indices[-1] + 1
    # Find first pointer-related token (pointer_start or pointer_pad)
    pointer_start_id = getattr(model.config, 'pointer_start_token_id', None)
    if pointer_start_id is not None:
        pointer_start_mask = (input_ids == pointer_start_id)
        ps_indices = torch.nonzero(pointer_start_mask, as_tuple=False).squeeze(-1)
        if len(ps_indices) > 0:
            query_end = ps_indices[0]
        else:
            query_end = target_indices[0]
    else:
        query_end = target_indices[0]
    
    if query_start >= query_end:
        # Fallback: use last few tokens before pointer_pad
        query_start_fallback = max(0, target_indices[0].item() - 20)
        query_end_fallback = target_indices[0].item()
        if query_start_fallback >= query_end_fallback:
            return None
        query_indices = torch.arange(query_start_fallback, query_end_fallback, device=device)
    else:
        query_indices = torch.arange(query_start, query_end, device=device)
    if len(query_indices) == 0:
        query_indices = torch.arange(max(0, target_indices[0].item() - 10), target_indices[0].item(), device=device)
    
    merged_indices = torch.cat([query_indices, target_indices], dim=0)
    
    # Do a fresh forward pass to get all hidden states
    full_input_ids = input_ids.unsqueeze(0)
    full_attn_mask = torch.ones_like(full_input_ids)
    
    with torch.no_grad():
        # Get position_ids
        position_ids, _ = model.get_rope_index(
            input_ids=full_input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=full_attn_mask,
        )
        
        # Forward pass with output_hidden_states
        outputs = model(
            input_ids=full_input_ids,
            attention_mask=full_attn_mask,
            pixel_values=prompt_inputs.get("pixel_values"),
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
    
    # hidden_states format: tuple of (n_layers+1) tensors, each (batch, seq_len, hidden_dim)
    # We need format [[layer0_hs, layer1_hs, ...]] for calculate_attention_from_qk
    hs_per_layer = list(outputs.hidden_states)  # list of (1, seq_len, d)
    
    # Calculate attention via QK recomputation
    calculated_attention = calculate_attention_from_qk(
        model=model,
        all_hidden_states=[hs_per_layer],  # wrap in list for timestep dimension
        all_position_ids=position_ids,
        query_indices=merged_indices,
        all_attention_mask=full_attn_mask,
    )
    
    # Cosine similarity for query weighting
    all_layer_hs = torch.stack(hs_per_layer[1:], dim=0)  # skip embedding layer, (n_layers, 1, seq, d)
    sample_layer_hs = all_layer_hs[:, 0, :, :]  # (n_layers, seq, d)
    query_hs = F.normalize(sample_layer_hs[:, query_indices, :], dim=-1)
    visual_hs = F.normalize(sample_layer_hs[:, visual_indices, :], dim=-1)
    sim_matrix = torch.einsum('lqd,lvd->lqv', query_hs, visual_hs)
    attn_per_query = sim_matrix.sum(dim=-1)
    
    topk_query_indices = None
    global_pattern_per_query = None
    
    if not getattr(model.config, 'kl_query_weighting', False):
        k = getattr(model.config, 'query_topk', 1)
        agg_attn = attn_per_query.sum(dim=0)
        _, topk_local_idx = torch.topk(agg_attn, min(k, len(query_indices)), largest=True)
        topk_query_indices = topk_local_idx
    else:
        global_pattern_per_query = attn_per_query.sum(dim=0).softmax(dim=-1)
    
    # Pointer head attention
    attn_scores, _ = model.multi_patch_pointer_head_attention(
        query_indices, visual_indices, target_indices,
        calculated_attention[0],
        topk_query_indices, global_pattern_per_query,
        batch_idx=0
    )
    
    # Attention map to coordinate
    merge_size = model.visual.spatial_merge_size
    _, n_height, n_width = (image_grid_thw[0] // merge_size).tolist()
    
    best_point, _, _, _ = get_prediction_region_point(
        attn_scores, n_width, n_height,
        return_all_regions=True, rect_center=False
    )
    
    return best_point  # (x, y) in [0, 1]


# ============================================================
# Data
# ============================================================

def load_grounding_dataset(data_path, image_folder, max_samples=None):
    """Load GUI grounding data in GRPO-compatible format."""
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
        
        # Extract bbox_gt and user instruction
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
        
        samples.append({
            "image_path": image_path,
            "instruction": user_text,
            "bbox_gt": bbox_gt,
        })
    
    print(f"Loaded {len(samples)} samples from {data_path}")
    return samples


# ============================================================
# Simplified GRPO Trainer for Attention-Based Grounding
# ============================================================

class FoveationGRPOTrainer:
    """
    GRPO trainer specialized for attention-based GUI grounding.
    
    Unlike standard GRPO that only uses text output for reward,
    this trainer extracts coordinates from the model's internal
    attention map (via ANCHOR/pointer_pad token) for reward computation.
    
    Simplified implementation without deep HuggingFace Trainer dependency,
    for clarity and easy debugging. Can be migrated to VLM-R1's 
    VLMGRPOTrainer later for multi-GPU support.
    """
    
    def __init__(
        self,
        model,
        processor,
        tokenizer,
        train_data,
        args,
        script_args,
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.args = args
        self.script_args = script_args
        
        # Reference model (for KL penalty)
        if args.beta > 0:
            import copy
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False
        else:
            self.ref_model = None
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=tokenizer.pad_token_id or 0,
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        
        # Metrics
        self.metrics = defaultdict(list)
    
    def _build_prompt(self, sample):
        """Build conversation prompt for a single sample."""
        # Modified system message to encourage thinking before pointing
        system_msg = (
            "You are a GUI grounding assistant. "
            "Given a screenshot and an instruction, first briefly describe where the target element is located, "
            "then click on it using pyautogui.click(<|pointer_pad|>). "
            "Example: The save button is in the top-right toolbar area. pyautogui.click(<|pointer_pad|>)"
        )
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_msg}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": sample["instruction"]},
                ]
            }
        ]
        
        # Tokenize
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, 
            add_generation_prompt=False,
            chat_template=chat_template
        )
        # Add generation prompt — encourage thinking before pointing
        # The model should generate free-form analysis, then pyautogui.click(<pointer_pad>)
        text += "<|im_start|>assistant<|recipient|>os\n"
        # Don't pre-fill anything — let the model generate both thinking and action
        
        image_inputs, _ = process_vision_info(conversation)
        inputs = self.processor(
            text=[text], images=image_inputs,
            return_tensors="pt", padding=True,
        )
        return inputs, image_inputs
    
    def _generate_completions(self, prompt_inputs, num_generations):
        """
        Generate multiple completions for a single prompt.
        
        After generation, append <pointer_pad> token at the end so the
        pointer head can extract attention-based coordinates.
        The model generates free-form text, then we force the ANCHOR token.
        """
        device = self.model.device
        completions = []
        
        pointer_pad_id = self.model.config.pointer_pad_token_id
        if isinstance(pointer_pad_id, list):
            pointer_pad_id = pointer_pad_id[0]
        
        for g in range(num_generations):
            inputs = {k: v.clone().to(device) for k, v in prompt_inputs.items()}
            
            # Generate free-form text (no pointer tokens needed)
            with torch.no_grad():
                result = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True,
                )
            
            prompt_len = inputs["input_ids"].shape[1]
            completion_ids = result.sequences[0, prompt_len:]
            
            # Append <pointer_pad> at the end of the sequence
            # This is the ANCHOR token — pointer head will use its hidden state
            # to attend over visual patches
            pointer_token = torch.tensor([pointer_pad_id], device=device)
            full_sequence_with_pointer = torch.cat([result.sequences[0], pointer_token])
            completion_ids_with_pointer = torch.cat([completion_ids, pointer_token])
            
            completions.append({
                "completion_ids": completion_ids,  # Original (without pointer) for log_prob
                "completion_ids_with_pointer": completion_ids_with_pointer,
                "full_sequence": full_sequence_with_pointer,  # With pointer for attention extraction
                "full_sequence_no_pointer": result.sequences[0],  # Without pointer for policy gradient
                "image_grid_thw": inputs.get("image_grid_thw"),
            })
        
        return completions
    
    def _compute_reward(self, completion_info, prompt_inputs, bbox_gt):
        """
        Compute reward for a single completion.
        
        1. Check format (has pointer_pad token)
        2. Do forward pass to get hidden states
        3. Extract coordinate from attention
        4. Gaussian point reward
        """
        device = self.model.device
        full_ids = completion_info["full_sequence"]
        
        # Format reward: check if pointer_pad was generated
        pointer_pad_id = self.model.config.pointer_pad_token_id
        if isinstance(pointer_pad_id, list):
            pointer_pad_id = pointer_pad_id[0]
        
        has_pointer = (full_ids == pointer_pad_id).any().item()
        if not has_pointer:
            return 0.0, None
        
        # Extract coordinate from attention (does a fresh forward pass)
        try:
            coord = extract_attention_coordinate(
                model=self.model,
                prompt_input_ids=full_ids,
                prompt_inputs=prompt_inputs,
                image_grid_thw=completion_info["image_grid_thw"],
            )
        except Exception as e:
            import traceback
            print(f"Attention extraction error: {e}")
            traceback.print_exc()
            return 0.0, None
        
        if coord is None:
            return 0.0, None
        
        pred_x, pred_y = coord
        
        # Gaussian point reward
        point_reward = gaussian_point_reward(
            pred_x, pred_y, bbox_gt, 
            alpha=self.script_args.reward_alpha
        )
        
        # Format reward bonus
        format_reward = 0.5  # bonus for generating valid pointer token
        
        total_reward = format_reward + point_reward
        
        return total_reward, coord
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, image_grid_thw=None):
        """Compute per-token log probabilities."""
        inputs = {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        }
        if image_grid_thw is not None:
            # Need pixel_values too — stored separately
            pass  # Will handle in full implementation
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits  # (1, seq_len, vocab)
        # Shift for autoregressive: predict token t from position t-1
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        token_log_probs = log_probs.gather(2, input_ids[:, 1:].unsqueeze(0).unsqueeze(-1)).squeeze(-1)
        
        return token_log_probs.squeeze(0)  # (seq_len - 1,)
    
    def train_step(self, batch_samples):
        """
        One GRPO training step:
        1. For each sample, generate G completions
        2. Compute rewards
        3. Compute advantages (group relative)
        4. Policy gradient update
        """
        device = self.model.device
        total_loss = 0.0
        n_samples = 0
        
        for sample in batch_samples:
            # Build prompt
            prompt_inputs, _ = self._build_prompt(sample)
            prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
            prompt_len = prompt_inputs["input_ids"].shape[1]
            
            # Generate G completions
            completions = self._generate_completions(
                prompt_inputs, self.args.num_generations
            )
            
            # Compute rewards
            rewards = []
            coords = []
            for comp in completions:
                r, c = self._compute_reward(comp, prompt_inputs, sample["bbox_gt"])
                rewards.append(r)
                coords.append(c)
            
            rewards_tensor = torch.tensor(rewards, device=device)
            
            # Group relative advantage
            mean_r = rewards_tensor.mean()
            std_r = rewards_tensor.std() + 1e-4
            advantages = (rewards_tensor - mean_r) / std_r
            
            # Policy gradient for each completion
            self.model.train()
            for comp, advantage in zip(completions, advantages):
                if advantage.abs() < 1e-6:
                    continue  # Skip zero advantage
                
                # Use the original generated sequence (without appended pointer) for policy gradient
                full_ids = comp["full_sequence_no_pointer"]
                completion_ids = comp["completion_ids"]
                comp_len = len(completion_ids)
                
                if comp_len == 0:
                    continue
                
                # Forward pass to get current log probs
                # We need pixel_values for the forward pass
                inputs_for_forward = {
                    k: v.to(device) for k, v in prompt_inputs.items()
                }
                
                # Construct full input_ids with completion
                full_input_ids = full_ids.unsqueeze(0).to(device)
                full_attention_mask = torch.ones_like(full_input_ids)
                
                outputs = self.model(
                    input_ids=full_input_ids,
                    attention_mask=full_attention_mask,
                    pixel_values=inputs_for_forward.get("pixel_values"),
                    image_grid_thw=inputs_for_forward.get("image_grid_thw"),
                )
                
                logits = outputs.logits  # (1, seq_len, vocab)
                
                # Get completion token log probs
                log_probs = F.log_softmax(logits[0, prompt_len-1:-1, :], dim=-1)
                token_ids = full_ids[prompt_len:].to(device)
                
                if len(token_ids) == 0 or log_probs.shape[0] == 0:
                    continue
                
                min_len = min(log_probs.shape[0], len(token_ids))
                per_token_logps = log_probs[:min_len].gather(
                    1, token_ids[:min_len].unsqueeze(-1)
                ).squeeze(-1)
                
                # GRPO loss: -advantage * log_prob (simplified, no clipping for v1)
                loss = -(advantage * per_token_logps.mean())
                
                # KL penalty
                if self.ref_model is not None and self.args.beta > 0:
                    with torch.no_grad():
                        ref_outputs = self.ref_model(
                            input_ids=full_input_ids,
                            attention_mask=full_attention_mask,
                            pixel_values=inputs_for_forward.get("pixel_values"),
                            image_grid_thw=inputs_for_forward.get("image_grid_thw"),
                        )
                    ref_log_probs = F.log_softmax(ref_outputs.logits[0, prompt_len-1:-1, :], dim=-1)
                    ref_per_token_logps = ref_log_probs[:min_len].gather(
                        1, token_ids[:min_len].unsqueeze(-1)
                    ).squeeze(-1)
                    kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                    loss = loss + self.args.beta * kl.mean()
                
                loss = loss / self.args.num_generations  # average over generations
                loss.backward()
                
                total_loss += loss.item()
                n_samples += 1
            
            # Log metrics
            self.metrics["reward"].append(mean_r.item())
            self.metrics["reward_std"].append(std_r.item())
            valid_coords = [c for c in coords if c is not None]
            if valid_coords:
                avg_dist = sum(
                    math.sqrt((c[0] - (sample["bbox_gt"][0]+sample["bbox_gt"][2])/2)**2 + 
                              (c[1] - (sample["bbox_gt"][1]+sample["bbox_gt"][3])/2)**2)
                    for c in valid_coords
                ) / len(valid_coords)
                self.metrics["avg_dist"].append(avg_dist)
        
        # Gradient step
        if n_samples > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return total_loss
    
    def train(self):
        """Main training loop."""
        device = self.model.device
        num_epochs = self.args.num_train_epochs
        batch_size = self.args.per_device_train_batch_size
        
        print(f"Starting GRPO training:")
        print(f"  Samples: {len(self.train_data)}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Generations per prompt: {self.args.num_generations}")
        print(f"  Temperature: {self.args.temperature}")
        print(f"  Beta (KL): {self.args.beta}")
        print(f"  Learning rate: {self.args.learning_rate}")
        
        global_step = 0
        
        for epoch in range(int(num_epochs)):
            random.shuffle(self.train_data)
            
            for i in range(0, len(self.train_data), batch_size):
                batch = self.train_data[i:i+batch_size]
                
                loss = self.train_step(batch)
                global_step += 1
                
                # Log
                if global_step % self.args.logging_steps == 0:
                    avg_metrics = {k: sum(v[-10:])/max(len(v[-10:]),1) for k, v in self.metrics.items()}
                    print(f"[Epoch {epoch+1}/{int(num_epochs)}] Step {global_step} | "
                          f"Loss: {loss:.4f} | "
                          f"Reward: {avg_metrics.get('reward', 0):.3f} | "
                          f"Dist: {avg_metrics.get('avg_dist', 0):.3f}")
                
                # Save checkpoint
                if global_step % self.args.save_steps == 0:
                    save_path = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    self.model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    print(f"Saved checkpoint to {save_path}")
        
        # Final save
        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        print(f"Training complete. Model saved to {self.args.output_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = transformers.HfArgumentParser((GRPOScriptArguments, GRPOTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    print(f"Loading model from {script_args.model_name_or_path}...")
    
    # Load model with pointer head architecture
    # If loading from base Qwen2.5-VL (no pointer weights), pointer head is randomly initialized
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        script_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        ignore_mismatched_sizes=True,  # Allow loading base model into pointer architecture
    )
    model.config.use_cache = False
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    
    # Unfreeze all parameters
    for p in model.parameters():
        p.requires_grad = True
    
    # Tokenizer & Processor
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    processor = AutoProcessor.from_pretrained(
        script_args.model_name_or_path,
        min_pixels=script_args.min_pixels,
        max_pixels=script_args.max_pixels,
    )
    processor.tokenizer = tokenizer
    
    # Add special tokens (pointer_start, pointer_end, pointer_pad, etc.)
    num_new = tokenizer.add_special_tokens(
        {"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS}
    )
    if num_new > 0:
        model.resize_token_embeddings(len(tokenizer))
        # Initialize new token embeddings with mean of existing
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings[-num_new:] = input_embeddings[:-num_new].mean(dim=0, keepdim=True)
        output_embeddings[-num_new:] = output_embeddings[:-num_new].mean(dim=0, keepdim=True)
    
    # Configure pointer token IDs in model config
    model.config.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
    model.config.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
    model.config.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
    # Set defaults for query weighting
    if not hasattr(model.config, 'query_topk'):
        model.config.query_topk = 1
    if not hasattr(model.config, 'kl_query_weighting'):
        model.config.kl_query_weighting = False
    if not hasattr(model.config, 'part_query_weighting'):
        model.config.part_query_weighting = False
    if not hasattr(model.config, 'layer_wise_query_weighting'):
        model.config.layer_wise_query_weighting = False
    
    print(f"pointer_pad_token_id: {model.config.pointer_pad_token_id}")
    print(f"Pointer head initialized: {hasattr(model, 'multi_patch_pointer_head_attention')}")
    
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_data = load_grounding_dataset(
        script_args.data_path,
        script_args.image_folder,
        max_samples=script_args.max_samples,
    )
    
    # Create output dir
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Train
    trainer = FoveationGRPOTrainer(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        train_data=train_data,
        args=training_args,
        script_args=script_args,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
