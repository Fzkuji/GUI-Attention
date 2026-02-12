"""
GRPO Training with Attention Sampling as Policy.

Key idea: Instead of argmax over attention map, we SAMPLE from the attention
distribution over visual patches. This gives us a proper log_prob for GRPO.

Policy = text generation (standard autoregressive) + attention sampling (pointer head)
Reward = format_reward(0.05) + position_reward(1/(relative_dist + 1))

Flow:
1. Model generates: "reasoning text... pyautogui.click(<pointer_start><pointer_pad><pointer_end>)"
   using GUI-AIMA's standard inference (use_placeholder=True)
2. From <pointer_pad>'s attention over visual patches → SAMPLE a patch position
3. Convert sampled patch to (x, y) coordinate
4. Compute reward based on distance to GT bbox
5. GRPO updates via text_log_prob + attention_sampling_log_prob

Usage:
    python train_grpo_attention.py \\
        --model_name_or_path /root/autodl-tmp/models/GUI-AIMA-3B \\
        --data_path /root/autodl-tmp/data/GUI-Actor/guiact_bbox.json \\
        --image_folder /root/autodl-tmp/data/GUI-Actor/images/GUIAct/web_imgs \\
        --output_dir /root/autodl-tmp/checkpoints/grpo_attn_sampling \\
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
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm

import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList,
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
    ForceFollowTokensLogitsProcessor,
    ForceFollowTokensLogitsProcessorSimple,
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
    format_reward_value: float = field(default=0.05, metadata={"help": "Reward for correct format"})
    use_placeholder: bool = field(default=True, metadata={"help": "Use placeholder inference mode"})


@dataclass
class GRPOTrainingArguments(transformers.TrainingArguments):
    # GRPO specific
    num_generations: int = field(default=8, metadata={"help": "Number of generations per prompt (G)"})
    max_completion_length: int = field(default=256, metadata={"help": "Max tokens to generate"})
    temperature: float = field(default=0.9, metadata={"help": "Sampling temperature for text generation"})
    attn_temperature: float = field(default=1.0, metadata={"help": "Temperature for attention sampling"})
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

def format_reward(completion_ids, model, value=0.05):
    """
    Check if the completion contains the proper pointer format:
    pyautogui.click(<pointer_start><pointer_pad><pointer_end>)
    Returns `value` if format is correct, 0 otherwise.
    """
    pointer_pad_id = model.config.pointer_pad_token_id
    if isinstance(pointer_pad_id, list):
        pad_ids_set = set(pointer_pad_id)
    else:
        pad_ids_set = {pointer_pad_id}

    has_pointer = any(tid in pad_ids_set for tid in completion_ids.tolist())
    return value if has_pointer else 0.0


def position_reward(pred_x, pred_y, bbox_gt, image_w_pixels, image_h_pixels):
    """
    Position reward: 1 / (relative_dist + 1)
    where relative_dist = dist_pixels / bbox_diag

    Args:
        pred_x, pred_y: predicted coords in [0,1] normalized space
        bbox_gt: [x1, y1, x2, y2] in [0,1] normalized space
        image_w_pixels, image_h_pixels: actual image size in pixels
    Returns:
        reward in (0, 1]
    """
    gt_cx = (bbox_gt[0] + bbox_gt[2]) / 2
    gt_cy = (bbox_gt[1] + bbox_gt[3]) / 2

    # Distance in pixels
    dist_pixels = math.sqrt(
        ((pred_x - gt_cx) * image_w_pixels) ** 2 +
        ((pred_y - gt_cy) * image_h_pixels) ** 2
    )

    # Bbox diagonal in pixels
    bbox_w = (bbox_gt[2] - bbox_gt[0]) * image_w_pixels
    bbox_h = (bbox_gt[3] - bbox_gt[1]) * image_h_pixels
    bbox_diag = math.sqrt(bbox_w ** 2 + bbox_h ** 2)
    bbox_diag = max(bbox_diag, 1.0)  # avoid div by zero

    relative_dist = dist_pixels / bbox_diag
    return 1.0 / (relative_dist + 1.0)


# ============================================================
# Attention Sampling
# ============================================================

def get_attention_distribution(
    model, input_ids, pixel_values, image_grid_thw, attention_mask,
):
    """
    Run a forward pass and extract the attention distribution over visual patches
    from the pointer head.

    Returns:
        attn_weights: (1, n_visual_patches) probability distribution
        visual_indices: indices of visual tokens in input_ids
        n_width, n_height: spatial dimensions of the attention map
        Or None if extraction fails.
    """
    device = input_ids.device

    # Find visual and pointer token indices
    visual_mask = (input_ids[0] == model.config.image_token_id)
    visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1)
    if visual_indices.numel() == 0:
        return None

    pointer_pad_id = model.config.pointer_pad_token_id
    if isinstance(pointer_pad_id, list):
        pointer_pad_mask = torch.isin(
            input_ids[0], torch.tensor(pointer_pad_id, device=device)
        )
    else:
        pointer_pad_mask = (input_ids[0] == pointer_pad_id)
    target_indices = torch.nonzero(pointer_pad_mask, as_tuple=False).squeeze(-1)
    if target_indices.numel() == 0:
        return None

    # Query indices: text tokens between last visual token and pointer_start
    query_start = visual_indices[-1].item() + 1
    ps_mask = (input_ids[0] == model.config.pointer_start_token_id)
    ps_positions = torch.nonzero(ps_mask, as_tuple=False).squeeze(-1)
    query_end = ps_positions[0].item() if ps_positions.numel() > 0 else target_indices[0].item()

    query_indices = torch.arange(query_start, query_end, device=device)
    if model.config.part_query_weighting and len(query_indices) > 12:
        query_indices = query_indices[:-12]
    if query_indices.numel() == 0:
        query_indices = torch.arange(
            max(0, target_indices[0].item() - 10), target_indices[0].item(), device=device
        )

    merged_indices = torch.cat([query_indices, target_indices], dim=0)

    # Forward pass with hidden states
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=None,
        attention_mask=attention_mask,
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        output_hidden_states=True,
    )

    # Wrap hidden states for calculate_attention_from_qk
    hs_per_layer = list(outputs.hidden_states)
    calculated_attention = calculate_attention_from_qk(
        model=model,
        all_hidden_states=[hs_per_layer],
        all_position_ids=position_ids,
        query_indices=merged_indices,
        all_attention_mask=attention_mask,
    )

    # Query importance weighting
    all_layer_hs = torch.stack(hs_per_layer[1:], dim=0)  # skip embedding layer
    sample_hs = all_layer_hs[:, 0, :, :]
    query_hs = F.normalize(sample_hs[:, query_indices, :], dim=-1)
    visual_hs = F.normalize(sample_hs[:, visual_indices, :], dim=-1)
    sim = torch.einsum('lqd,lvd->lqv', query_hs, visual_hs)
    attn_per_query = sim.sum(dim=-1)

    topk_query_indices = None
    global_pattern = None

    if not getattr(model.config, 'kl_query_weighting', False):
        k = getattr(model.config, 'query_topk', 1)
        agg = attn_per_query.sum(dim=0)
        _, topk_query_indices = torch.topk(agg, min(k, len(query_indices)), largest=True)
    else:
        global_pattern = attn_per_query.sum(dim=0).softmax(dim=-1)

    # Pointer head → attention distribution over visual patches
    attn_weights, _ = model.multi_patch_pointer_head_attention(
        query_indices, visual_indices, target_indices,
        calculated_attention[0],
        topk_query_indices, global_pattern,
        batch_idx=0,
    )

    # attn_weights: (1, n_visual_patches) — already normalized (sums to 1)
    merge_size = model.visual.spatial_merge_size
    _, n_height, n_width = (image_grid_thw[0] // merge_size).tolist()

    return {
        "attn_weights": attn_weights,  # (1, n_visual)
        "n_width": int(n_width),
        "n_height": int(n_height),
        "outputs": outputs,  # for getting text log_probs
    }


def sample_coordinate_from_attention(attn_weights, n_width, n_height, temperature=1.0):
    """
    Sample a patch index from the attention distribution and convert to (x, y).

    Args:
        attn_weights: (1, n_visual) probability distribution
        n_width, n_height: spatial grid dimensions
        temperature: sampling temperature (1.0 = standard, <1 = sharper, >1 = flatter)

    Returns:
        pred_x, pred_y: sampled coordinates in [0, 1]
        log_prob: log probability of the sampled patch
        sampled_idx: the sampled patch index
    """
    # Apply temperature
    if temperature != 1.0:
        logits = torch.log(attn_weights.clamp(min=1e-10)) / temperature
        probs = F.softmax(logits, dim=-1)
    else:
        probs = attn_weights

    # Sample from the distribution
    dist = torch.distributions.Categorical(probs=probs.squeeze(0))
    sampled_idx = dist.sample()
    log_prob = dist.log_prob(sampled_idx)

    # Convert patch index to (x, y) coordinates
    patch_y = sampled_idx.item() // n_width
    patch_x = sampled_idx.item() % n_width
    pred_x = (patch_x + 0.5) / n_width
    pred_y = (patch_y + 0.5) / n_height

    return pred_x, pred_y, log_prob, sampled_idx.item()


# ============================================================
# Data
# ============================================================

def load_grounding_dataset(data_path, image_folder, max_samples=None):
    """Load GUI grounding data."""
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

        samples.append({
            "image_path": image_path,
            "instruction": user_text,
            "bbox_gt": bbox_gt,
        })

    print(f"Loaded {len(samples)} samples from {data_path}")
    return samples


# ============================================================
# GRPO Trainer with Attention Sampling
# ============================================================

class AttentionSamplingGRPOTrainer:
    """
    GRPO trainer where the policy is:
      - Text generation (standard LM) → text_log_probs
      - Attention sampling (pointer head) → attn_log_prob

    Total log_prob = mean(text_token_log_probs) + attn_log_prob

    Different samples from attention → different coordinates → different rewards
    → GRPO has variance to learn from.
    """

    def __init__(self, model, ref_model, processor, tokenizer, train_data, args, script_args):
        self.model = model
        self.ref_model = ref_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.args = args
        self.script_args = script_args

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

        # LR scheduler
        total_steps = (
            int(args.num_train_epochs) *
            len(train_data) //
            max(args.per_device_train_batch_size, 1) //
            max(args.gradient_accumulation_steps, 1)
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7
        )

        self.metrics = defaultdict(list)
        self.global_step = 0

        # Number of pointer points
        if isinstance(model.config.pointer_pad_token_id, list):
            self.n_points = len(model.config.pointer_pad_token_id)
        else:
            self.n_points = 1

    def _build_prompt_inputs(self, sample):
        """Build tokenized prompt for a single sample using GUI-AIMA format."""
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": grounding_system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": sample["instruction"]},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False,
            add_generation_prompt=False,
            chat_template=chat_template,
        )

        if self.script_args.use_placeholder:
            # Standard GUI-AIMA placeholder format
            text += "<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
        else:
            text += "<|im_start|>assistant<|recipient|>os\n"

        image_inputs, _ = process_vision_info(conversation)
        inputs = self.processor(
            text=[text], images=image_inputs,
            return_tensors="pt", padding=True,
        )
        return inputs

    def _generate_and_sample(self, prompt_inputs, sample, num_generations):
        """
        For each generation:
        1. If use_placeholder: no text generation needed, just run forward pass
           to get attention distribution, then SAMPLE from it.
        2. If not use_placeholder: generate text, then extract attention + sample.

        Returns list of dicts with reward, log_probs, etc.
        """
        device = self.model.device
        inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        # Get image size for reward calculation
        try:
            img = Image.open(sample["image_path"])
            img_w, img_h = img.size
        except Exception:
            img_w, img_h = 1920, 1080  # fallback

        generations = []

        if self.script_args.use_placeholder:
            # With placeholder, the full sequence is already in input_ids.
            # We just need to run forward to get attention, then sample multiple times.
            # The text part is fixed (no text generation variance), but attention
            # sampling provides the stochasticity for GRPO.

            self.model.eval()
            with torch.no_grad():
                attn_result = get_attention_distribution(
                    self.model,
                    inputs["input_ids"],
                    inputs.get("pixel_values"),
                    inputs.get("image_grid_thw"),
                    inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
                )

            if attn_result is None:
                # Failed to extract attention, return zero-reward generations
                for _ in range(num_generations):
                    generations.append({
                        "reward": 0.0,
                        "attn_log_prob": torch.tensor(0.0, device=device),
                        "text_log_probs": None,
                        "pred_coord": None,
                        "input_ids": inputs["input_ids"][0],
                    })
                return generations

            attn_weights = attn_result["attn_weights"]
            n_w = attn_result["n_width"]
            n_h = attn_result["n_height"]

            for _ in range(num_generations):
                # Sample from attention distribution
                pred_x, pred_y, attn_log_prob, _ = sample_coordinate_from_attention(
                    attn_weights, n_w, n_h,
                    temperature=self.args.attn_temperature,
                )

                # Compute reward
                fmt_r = self.script_args.format_reward_value  # always has correct format with placeholder
                pos_r = position_reward(pred_x, pred_y, sample["bbox_gt"], img_w, img_h)
                total_r = fmt_r + pos_r

                generations.append({
                    "reward": total_r,
                    "attn_log_prob": attn_log_prob,
                    "text_log_probs": None,  # no text variance with placeholder
                    "pred_coord": (pred_x, pred_y),
                    "input_ids": inputs["input_ids"][0],
                    "prompt_len": prompt_len,
                })

        else:
            # Without placeholder: generate text completions, then extract attention
            logits_processor = ForceFollowTokensLogitsProcessor(
                tokenizer=self.tokenizer,
                number_of_points=self.n_points,
            )

            for _ in range(num_generations):
                # Reset logits processor state
                lp = ForceFollowTokensLogitsProcessor(
                    tokenizer=self.tokenizer,
                    number_of_points=self.n_points,
                )

                self.model.eval()
                with torch.no_grad():
                    result = self.model.generate(
                        **{k: v.clone() for k, v in inputs.items()},
                        generation_config=self.generation_config,
                        logits_processor=LogitsProcessorList([lp]),
                        return_dict_in_generate=True,
                    )

                full_ids = result.sequences[0]  # (seq_len,)
                completion_ids = full_ids[prompt_len:]

                # Check format
                fmt_r = format_reward(completion_ids, self.model, self.script_args.format_reward_value)

                # Extract attention from the full sequence
                full_input_ids = full_ids.unsqueeze(0)
                full_attn_mask = torch.ones_like(full_input_ids)

                with torch.no_grad():
                    attn_result = get_attention_distribution(
                        self.model,
                        full_input_ids,
                        inputs.get("pixel_values"),
                        inputs.get("image_grid_thw"),
                        full_attn_mask,
                    )

                if attn_result is None:
                    generations.append({
                        "reward": fmt_r,
                        "attn_log_prob": torch.tensor(0.0, device=device),
                        "text_log_probs": None,
                        "pred_coord": None,
                        "input_ids": full_ids,
                        "completion_ids": completion_ids,
                        "prompt_len": prompt_len,
                    })
                    continue

                # Sample coordinate
                pred_x, pred_y, attn_log_prob, _ = sample_coordinate_from_attention(
                    attn_result["attn_weights"],
                    attn_result["n_width"],
                    attn_result["n_height"],
                    temperature=self.args.attn_temperature,
                )

                pos_r = position_reward(pred_x, pred_y, sample["bbox_gt"], img_w, img_h)
                total_r = fmt_r + pos_r

                generations.append({
                    "reward": total_r,
                    "attn_log_prob": attn_log_prob,
                    "text_log_probs": None,  # computed during training forward pass
                    "pred_coord": (pred_x, pred_y),
                    "input_ids": full_ids,
                    "completion_ids": completion_ids,
                    "prompt_len": prompt_len,
                })

        return generations

    def _compute_grpo_loss(self, generations, prompt_inputs):
        """
        Compute GRPO loss from a group of generations.

        For placeholder mode (no text generation):
            loss = -advantage * attn_log_prob

        For free generation mode:
            loss = -advantage * (mean_text_log_prob + attn_log_prob)
                 + beta * KL(ref || policy)
        """
        device = self.model.device
        rewards = torch.tensor([g["reward"] for g in generations], device=device)

        # Group-relative advantage
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0

        if self.script_args.use_placeholder:
            # Placeholder mode: need differentiable attention log_probs
            # Re-run forward pass WITH gradients to get differentiable attention
            inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

            self.model.train()
            attn_result = get_attention_distribution(
                self.model,
                inputs["input_ids"],
                inputs.get("pixel_values"),
                inputs.get("image_grid_thw"),
                inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            )

            if attn_result is None:
                return torch.tensor(0.0, device=device, requires_grad=True), 0

            attn_weights = attn_result["attn_weights"]  # (1, n_visual)
            n_w = attn_result["n_width"]
            n_h = attn_result["n_height"]

            # For each generation, compute differentiable log_prob of sampled patch
            for gen, adv in zip(generations, advantages):
                if gen["pred_coord"] is None or abs(adv.item()) < 1e-8:
                    continue

                pred_x, pred_y = gen["pred_coord"]
                # Recover the sampled patch index from coordinates
                patch_x = int(pred_x * n_w - 0.5 + 0.5)  # round to nearest
                patch_y = int(pred_y * n_h - 0.5 + 0.5)
                patch_x = max(0, min(patch_x, n_w - 1))
                patch_y = max(0, min(patch_y, n_h - 1))
                sampled_idx = patch_y * n_w + patch_x

                # Apply temperature
                if self.args.attn_temperature != 1.0:
                    logits = torch.log(attn_weights.clamp(min=1e-10)) / self.args.attn_temperature
                    log_probs = F.log_softmax(logits, dim=-1)
                else:
                    log_probs = torch.log(attn_weights.clamp(min=1e-10))

                attn_log_prob = log_probs[0, sampled_idx]

                # GRPO loss: -advantage * log_prob
                loss_i = -adv * attn_log_prob
                total_loss = total_loss + loss_i
                n_valid += 1

        else:
            # Free generation mode: need text log_probs + attention log_probs
            inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

            for gen, adv in zip(generations, advantages):
                if gen["pred_coord"] is None or abs(adv.item()) < 1e-8:
                    continue

                full_ids = gen["input_ids"].unsqueeze(0).to(device)
                full_attn_mask = torch.ones_like(full_ids)
                prompt_len = gen["prompt_len"]
                completion_ids = gen.get("completion_ids")
                if completion_ids is None or len(completion_ids) == 0:
                    continue

                # Forward pass for text log_probs (with gradients)
                self.model.train()
                outputs = self.model(
                    input_ids=full_ids,
                    attention_mask=full_attn_mask,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    output_hidden_states=True,
                )

                logits = outputs.logits[0]  # (seq_len, vocab)
                # Text log_probs for completion tokens
                log_probs = F.log_softmax(logits[prompt_len - 1:-1, :], dim=-1)
                token_ids = full_ids[0, prompt_len:]
                min_len = min(log_probs.shape[0], len(token_ids))
                text_log_probs = log_probs[:min_len].gather(
                    1, token_ids[:min_len].unsqueeze(-1)
                ).squeeze(-1)
                mean_text_lp = text_log_probs.mean()

                # Attention log_prob (differentiable)
                # Re-extract attention from this forward pass
                attn_result = get_attention_distribution(
                    self.model, full_ids,
                    inputs.get("pixel_values"),
                    inputs.get("image_grid_thw"),
                    full_attn_mask,
                )

                if attn_result is not None:
                    pred_x, pred_y = gen["pred_coord"]
                    n_w = attn_result["n_width"]
                    n_h = attn_result["n_height"]
                    patch_x = max(0, min(int(pred_x * n_w), n_w - 1))
                    patch_y = max(0, min(int(pred_y * n_h), n_h - 1))
                    sampled_idx = patch_y * n_w + patch_x

                    attn_lp = torch.log(attn_result["attn_weights"][0, sampled_idx].clamp(min=1e-10))
                    combined_lp = mean_text_lp + attn_lp
                else:
                    combined_lp = mean_text_lp

                # KL penalty
                kl_penalty = torch.tensor(0.0, device=device)
                if self.ref_model is not None and self.args.beta > 0:
                    with torch.no_grad():
                        ref_outputs = self.ref_model(
                            input_ids=full_ids,
                            attention_mask=full_attn_mask,
                            pixel_values=inputs.get("pixel_values"),
                            image_grid_thw=inputs.get("image_grid_thw"),
                        )
                    ref_log_probs = F.log_softmax(
                        ref_outputs.logits[0, prompt_len - 1:-1, :], dim=-1
                    )
                    ref_text_lp = ref_log_probs[:min_len].gather(
                        1, token_ids[:min_len].unsqueeze(-1)
                    ).squeeze(-1)
                    # Approximate KL: E_pi[log(pi/ref)] ≈ mean(log_pi - log_ref)
                    kl_penalty = (text_log_probs - ref_text_lp).mean()

                loss_i = -adv * combined_lp + self.args.beta * kl_penalty
                total_loss = total_loss + loss_i
                n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid

        return total_loss, n_valid

    def train_step(self, batch_samples):
        """One GRPO step over a batch of samples."""
        device = self.model.device
        accumulated_loss = 0.0
        accumulated_n = 0

        for sample in batch_samples:
            try:
                # Build prompt
                prompt_inputs = self._build_prompt_inputs(sample)

                # Generate + sample attention (no gradients)
                generations = self._generate_and_sample(
                    prompt_inputs, sample, self.args.num_generations
                )

                # Compute GRPO loss (with gradients)
                loss, n_valid = self._compute_grpo_loss(generations, prompt_inputs)

                if n_valid > 0 and loss.requires_grad:
                    scaled_loss = loss / max(self.args.gradient_accumulation_steps, 1)
                    scaled_loss.backward()
                    accumulated_loss += loss.item()
                    accumulated_n += 1

                # Log metrics
                rewards = [g["reward"] for g in generations]
                self.metrics["reward"].append(sum(rewards) / len(rewards))
                valid_coords = [g["pred_coord"] for g in generations if g["pred_coord"] is not None]
                if valid_coords:
                    gt_cx = (sample["bbox_gt"][0] + sample["bbox_gt"][2]) / 2
                    gt_cy = (sample["bbox_gt"][1] + sample["bbox_gt"][3]) / 2
                    avg_dist = sum(
                        math.sqrt((c[0] - gt_cx) ** 2 + (c[1] - gt_cy) ** 2)
                        for c in valid_coords
                    ) / len(valid_coords)
                    self.metrics["avg_dist"].append(avg_dist)

                    # Check bbox hit rate
                    hits = sum(
                        1 for c in valid_coords
                        if sample["bbox_gt"][0] <= c[0] <= sample["bbox_gt"][2]
                        and sample["bbox_gt"][1] <= c[1] <= sample["bbox_gt"][3]
                    )
                    self.metrics["hit_rate"].append(hits / len(valid_coords))

            except Exception as e:
                import traceback
                print(f"Error processing sample: {e}")
                traceback.print_exc()
                continue

        return accumulated_loss, accumulated_n

    def train(self):
        """Main training loop."""
        device = self.model.device
        num_epochs = int(self.args.num_train_epochs)
        batch_size = self.args.per_device_train_batch_size
        grad_accum = max(self.args.gradient_accumulation_steps, 1)

        print(f"=== Attention Sampling GRPO Training ===")
        print(f"  Samples: {len(self.train_data)}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Effective batch: {batch_size * grad_accum}")
        print(f"  Generations per prompt: {self.args.num_generations}")
        print(f"  Text temperature: {self.args.temperature}")
        print(f"  Attention temperature: {self.args.attn_temperature}")
        print(f"  Beta (KL): {self.args.beta}")
        print(f"  Use placeholder: {self.script_args.use_placeholder}")
        print(f"  Learning rate: {self.args.learning_rate}")

        self.optimizer.zero_grad()
        micro_step = 0

        for epoch in range(num_epochs):
            random.shuffle(self.train_data)
            pbar = tqdm(range(0, len(self.train_data), batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}")

            for i in pbar:
                batch = self.train_data[i:i + batch_size]
                loss, n_valid = self.train_step(batch)
                micro_step += 1

                # Gradient step
                if micro_step % grad_accum == 0:
                    if any(p.grad is not None for p in self.model.parameters()):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Log
                    if self.global_step % self.args.logging_steps == 0:
                        avg_r = sum(self.metrics["reward"][-20:]) / max(len(self.metrics["reward"][-20:]), 1)
                        avg_d = sum(self.metrics["avg_dist"][-20:]) / max(len(self.metrics["avg_dist"][-20:]), 1)
                        avg_h = sum(self.metrics["hit_rate"][-20:]) / max(len(self.metrics["hit_rate"][-20:]), 1)
                        pbar.set_postfix(
                            step=self.global_step, loss=f"{loss:.4f}",
                            reward=f"{avg_r:.3f}", dist=f"{avg_d:.4f}", hit=f"{avg_h:.2%}"
                        )
                        print(f"  [Step {self.global_step}] loss={loss:.4f} reward={avg_r:.3f} "
                              f"dist={avg_d:.4f} hit_rate={avg_h:.2%} lr={self.scheduler.get_last_lr()[0]:.2e}")

                    # Save checkpoint
                    if self.global_step % self.args.save_steps == 0:
                        self._save_checkpoint(f"checkpoint-{self.global_step}")

        # Final save
        self._save_checkpoint("final")
        print(f"Training complete. Model saved to {self.args.output_dir}")

    def _save_checkpoint(self, name):
        save_path = os.path.join(self.args.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        # Save metrics
        with open(os.path.join(save_path, "metrics.json"), "w") as f:
            json.dump({k: v[-100:] for k, v in self.metrics.items()}, f)
        print(f"  Saved checkpoint: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = transformers.HfArgumentParser((GRPOScriptArguments, GRPOTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Loading model from {script_args.model_name_or_path}...")

    # Load model
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        script_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

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

    # Add special tokens
    num_new = tokenizer.add_special_tokens(
        {"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS}
    )
    if num_new > 0:
        model.resize_token_embeddings(len(tokenizer))
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings[-num_new:] = input_embeddings[:-num_new].mean(dim=0, keepdim=True)
        output_embeddings[-num_new:] = output_embeddings[:-num_new].mean(dim=0, keepdim=True)

    # Configure pointer token IDs
    model.config.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
    model.config.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
    model.config.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
    if not hasattr(model.config, 'query_topk'):
        model.config.query_topk = 1
    if not hasattr(model.config, 'kl_query_weighting'):
        model.config.kl_query_weighting = False
    if not hasattr(model.config, 'part_query_weighting'):
        model.config.part_query_weighting = False
    if not hasattr(model.config, 'layer_wise_query_weighting'):
        model.config.layer_wise_query_weighting = False

    print(f"pointer_pad_token_id: {model.config.pointer_pad_token_id}")
    print(f"Pointer head: {hasattr(model, 'multi_patch_pointer_head_attention')}")

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Reference model for KL penalty
    ref_model = None
    if training_args.beta > 0 and not script_args.use_placeholder:
        import copy
        print("Creating reference model for KL penalty...")
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    # Load data
    train_data = load_grounding_dataset(
        script_args.data_path,
        script_args.image_folder,
        max_samples=script_args.max_samples,
    )

    os.makedirs(training_args.output_dir, exist_ok=True)

    # Train
    trainer = AttentionSamplingGRPOTrainer(
        model=model,
        ref_model=ref_model,
        processor=processor,
        tokenizer=tokenizer,
        train_data=train_data,
        args=training_args,
        script_args=script_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
