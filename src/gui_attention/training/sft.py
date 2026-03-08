"""
Saccade Foveation Training: Dual Head (LookHead + ClickHead).

Uses Qwen2.5-VL + peft LoRA + DualActionHead. DDP via torchrun.

Multi-round training from step 0 (no single-round warmup):
  Round 0: low-res full image → LookHead → KL loss on GT region → crop
  Round 1+: low-res + crops → LookHead → saccade or stop
  When GT is in crop → break → ClickHead on ALL crop tokens (after click_phase_step)

Usage:
    python -m gui_attention.train \
        --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
        --data_path /path/to/guiact_bbox.json \
        --image_folder /path/to/images \
        --output_dir /path/to/checkpoints
"""

import json
import math
import os
import random
import re
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from PIL import Image, ImageFile
from tqdm import tqdm

from gui_attention.modeling.attention import (
    extract_anchor_hidden_states,
    extract_visual_hidden_states,
    identify_attended_image,
    token_to_spatial,
)
from gui_attention.inputs.builder import MultiRoundInputBuilder
from gui_attention.inputs.crop import crop_image, crop_image_bbox, point_in_bbox
from gui_attention.inputs.labels import compute_binary_labels, compute_overlap_mask
from gui_attention.constants import (
    DEFAULT_LOOK_PAD_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
    HIGH_RES_MAX_PIXELS,
    IGNORE_INDEX,
    LOW_RES_MAX_PIXELS,
)
from gui_attention.runtime.proposals import box_center, select_top_proposal_bbox
from gui_attention.modeling.model import Qwen25VLWithDualHead, build_model


ImageFile.LOAD_TRUNCATED_IMAGES = True


# -- LM Labels ---------------------------------------------------------------

def make_lm_labels_and_weights(
    input_ids,
    tokenizer,
    round_idx=0,
    *,
    reasoning_weight=0.1,
    format_weight=0.5,
    look_weight=1.0,
    click_weight=2.0,
):
    """Create per-round LM labels plus token weights for weighted CE."""
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    weights = torch.zeros_like(input_ids, dtype=torch.float32)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    pointer_start_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_START_TOKEN)
    pointer_end_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_END_TOKEN)
    look_pad_id = tokenizer.convert_tokens_to_ids(DEFAULT_LOOK_PAD_TOKEN)
    click_pad_id = tokenizer.convert_tokens_to_ids(DEFAULT_POINTER_PAD_TOKEN)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    ids = input_ids[0].tolist()
    seq_len = len(ids)

    assistant_turns = []
    i = 0
    while i < seq_len:
        if ids[i] == im_start_id:
            end_check = min(i + 4, seq_len)
            role_text = tokenizer.decode(ids[i + 1:end_check], skip_special_tokens=False)
            if role_text.lstrip().startswith("assistant"):
                content_start = i + 1
                j = i + 1
                while j < seq_len and ids[j] != im_start_id:
                    j += 1
                content_end = j
                assistant_turns.append((content_start, content_end))
                i = j
                continue
        i += 1

    if round_idx < len(assistant_turns):
        start, end = assistant_turns[round_idx]
        labels[0, start:end] = input_ids[0, start:end]
        weights[0, start:end] = reasoning_weight

        format_token_ids = {
            tid for tid in (pointer_start_id, pointer_end_id, im_end_id, eos_id)
            if tid is not None and tid >= 0
        }
        for tid in format_token_ids:
            weights[input_ids == tid] = format_weight
        if look_pad_id is not None and look_pad_id >= 0:
            weights[input_ids == look_pad_id] = look_weight
        if click_pad_id is not None and click_pad_id >= 0:
            weights[input_ids == click_pad_id] = click_weight

    weights = weights * (labels != IGNORE_INDEX).float()
    return labels, weights


def compute_weighted_lm_loss(logits, labels, weights):
    """Weighted token-level CE on the current round's assistant response."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = weights[..., 1:].contiguous()

    valid_mask = shift_labels != IGNORE_INDEX
    if not valid_mask.any():
        return None

    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view_as(shift_labels)

    denom = shift_weights[valid_mask].sum().clamp_min(1e-6)
    return (per_token[valid_mask] * shift_weights[valid_mask]).sum() / denom


# -- Arguments ----------------------------------------------------------------

@dataclass
class ScriptArgs:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    data_path: str = field(default=None, metadata={"help": "JSON path(s), comma-separated for multiple datasets"})
    image_folder: str = field(default=None, metadata={"help": "Image folder(s), comma-separated, matching data_path order"})
    max_samples: Optional[int] = field(default=None)
    min_pixels: int = field(default=3136)
    low_res_max_pixels: int = field(default=LOW_RES_MAX_PIXELS)
    high_res_max_pixels: int = field(default=HIGH_RES_MAX_PIXELS)
    crop_size: int = field(default=308, metadata={"help": "Fixed crop side length in pixels. Must be divisible by 28."})
    crop_upscale: int = field(default=3, metadata={"help": "Integer upscale factor for fixed crop."})
    crop_jitter: float = field(default=0.05, metadata={"help": "Random jitter for crop center (fraction of image)"})
    max_saccade_rounds: int = field(default=6, metadata={"help": "Max rounds per sample (round 0 + up to N-1 crop saccades)."})
    click_phase_step: int = field(default=3000, metadata={"help": "Step to start training ClickHead. Before this, only LookHead trains."})
    use_dual_tokens: bool = field(default=False, metadata={"help": "Use <look_pad>/<click_pad> dual tokens instead of <pointer_pad>."})
    free_reasoning_sft: bool = field(default=True, metadata={"help": "Bootstrap SFT with brief reasoning + action span outputs instead of the old fixed assistant string."})
    append_assistant_eos: bool = field(default=True, metadata={"help": "Append tokenizer EOS after supervised assistant turns to teach the model where to stop."})
    # Loss weights
    lm_loss_weight: float = field(default=0.5, metadata={"help": "Weight for LM (next-token prediction) loss."})
    look_loss_weight: float = field(default=1.0, metadata={"help": "Weight for LookHead KL loss."})
    click_loss_weight: float = field(default=4.0, metadata={"help": "Weight for ClickHead KL loss."})
    lm_reasoning_token_weight: float = field(default=0.1, metadata={"help": "Weight for ordinary reasoning tokens in LM loss."})
    lm_format_token_weight: float = field(default=0.5, metadata={"help": "Weight for pointer boundary and EOS tokens in LM loss."})
    lm_look_token_weight: float = field(default=1.0, metadata={"help": "Weight for <look_pad> in LM loss."})
    lm_click_token_weight: float = field(default=2.0, metadata={"help": "Weight for <pointer_pad> click token in LM loss."})
    # LoRA
    use_lora: bool = field(default=True)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_target_modules: str = field(default="q_proj,v_proj")
    # LR
    action_head_lr: float = field(default=1e-4, metadata={"help": "LR for dual head (both LookHead and ClickHead)"})
    lora_lr: float = field(default=5e-5, metadata={"help": "LR for backbone params (LoRA or full)"})
    # Per-dataset sample limits
    max_samples_per_dataset: Optional[str] = field(default=None, metadata={
        "help": "Comma-separated per-dataset max samples, e.g. '0,0,0,60000,0'. 0=no limit."
    })
    # Dual-head initialization from GUI-Actor pointer head
    click_head_from: Optional[str] = field(default=None, metadata={
        "help": "Path to GUI-Actor checkpoint (safetensors dir) to initialize both LookHead and ClickHead from pointer_head weights"
    })
    # Resume
    init_ckpt: Optional[str] = field(default=None, metadata={"help": "Checkpoint dir to initialize model weights from without resuming optimizer/step state"})
    resume_ckpt: Optional[str] = field(default=None, metadata={"help": "Checkpoint dir to resume training from"})


@dataclass
class TrainArgs(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    gradient_checkpointing: bool = field(default=True)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)


# -- Data ---------------------------------------------------------------------

def load_single_dataset(data_path, image_folder):
    with open(data_path) as f:
        raw = json.load(f)
    samples = []
    for item in raw:
        img_file = item["image"]
        if isinstance(img_file, list):
            img_file = img_file[0]
        img_path = os.path.join(image_folder, img_file)
        if not os.path.exists(img_path):
            continue
        bbox_gt = None
        user_text = ""
        for conv in item["conversations"]:
            if conv.get("bbox_gt") is not None:
                bbox_gt = conv["bbox_gt"]
            role = conv.get("from", conv.get("role", ""))
            if role in ("human", "user"):
                user_text = re.sub(r"<image>", "", conv.get("value", conv.get("content", ""))).strip()
        if bbox_gt and user_text:
            samples.append({"image_path": img_path, "instruction": user_text, "bbox_gt": bbox_gt})
    return samples


def load_dataset(data_path, image_folder, max_samples=None, max_samples_per_dataset=None):
    """Load one or more datasets. Comma-separated paths for multiple."""
    data_paths = [p.strip() for p in data_path.split(",")]
    image_folders = [p.strip() for p in image_folder.split(",")]
    if len(image_folders) == 1 and len(data_paths) > 1:
        image_folders = image_folders * len(data_paths)
    assert len(data_paths) == len(image_folders)

    per_ds_limits = None
    if max_samples_per_dataset:
        per_ds_limits = [int(x.strip()) for x in max_samples_per_dataset.split(",")]
        assert len(per_ds_limits) == len(data_paths)

    samples = []
    for i, (dp, imf) in enumerate(zip(data_paths, image_folders)):
        s = load_single_dataset(dp, imf)
        limit = per_ds_limits[i] if per_ds_limits else 0
        if limit > 0 and len(s) > limit:
            s = s[:limit]
            print(f"  {os.path.basename(dp)}: {len(s)} samples (limited to {limit})")
        else:
            print(f"  {os.path.basename(dp)}: {len(s)} samples")
        samples.extend(s)

    if max_samples:
        random.shuffle(samples)
        samples = samples[:max_samples]
    print(f"Total: {len(samples)} samples")
    return samples


# -- Helpers ------------------------------------------------------------------

def _get_model_device(model):
    if hasattr(model, 'device'):
        return model.device
    return next(model.parameters()).device


def _get_backbone_for_rope(model):
    """Walk through wrapping layers to find get_rope_index."""
    backbone = model.backbone
    for _attr in ('base_model', 'model', 'model'):
        if hasattr(backbone, 'get_rope_index'):
            return backbone
        if hasattr(backbone, _attr):
            backbone = getattr(backbone, _attr)
    if not hasattr(backbone, 'get_rope_index'):
        raise AttributeError(f"Cannot find get_rope_index. Final type: {type(backbone)}")
    return backbone


def _get_visual_module(model):
    """Find the visual encoder module."""
    _backbone = model.backbone
    if hasattr(_backbone, 'base_model') and hasattr(_backbone.base_model, 'model'):
        _inner = _backbone.base_model.model
    else:
        _inner = _backbone
    if hasattr(_inner, 'visual'):
        return _inner.visual
    elif hasattr(_inner, 'model') and hasattr(_inner.model, 'visual'):
        return _inner.model.visual
    raise AttributeError(f"Cannot find visual module in {type(_inner)}")


# -- Trainer ------------------------------------------------------------------

class SaccadeTrainer:
    def __init__(self, model: Qwen25VLWithDualHead, tokenizer, train_data,
                 args: TrainArgs, script_args: ScriptArgs):
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.args = args
        self.sa = script_args
        self.builder = MultiRoundInputBuilder(
            script_args.model_name_or_path, tokenizer, script_args.min_pixels,
            low_res_max_pixels=script_args.low_res_max_pixels,
            high_res_max_pixels=script_args.high_res_max_pixels,
        )

        self.model = model

        # Distributed
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Total steps for scheduler
        total_steps = (
            int(args.num_train_epochs) * len(train_data)
            // self.world_size
            // max(args.per_device_train_batch_size, 1)
            // max(args.gradient_accumulation_steps, 1)
        )
        if args.max_steps and args.max_steps > 0:
            total_steps = min(total_steps, args.max_steps)

        # Dual LR: heads get higher LR, backbone gets lower LR
        head_params = list(model.dual_head.parameters())
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        param_groups = [
            {"params": head_params, "lr": script_args.action_head_lr},
            {"params": backbone_params, "lr": script_args.lora_lr},
        ]
        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=args.weight_decay,
        )
        warmup_steps = int(total_steps * args.warmup_ratio) if args.warmup_ratio > 0 else 0
        self.scheduler = transformers.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(total_steps, 1),
        )

        self.metrics = defaultdict(list)
        self.global_step = 0

        # Resume
        if script_args.resume_ckpt:
            state_path = os.path.join(script_args.resume_ckpt, "training_state.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu")
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
                self.global_step = state["global_step"]
                if self.rank == 0:
                    print(f"  Resumed training state: global_step={self.global_step}")

    # -- helpers -------------------------------------------------------------

    @property
    def in_click_phase(self):
        """Whether ClickHead is active."""
        return self.global_step >= self.sa.click_phase_step

    @property
    def max_rounds(self):
        return self.sa.max_saccade_rounds

    # -- multi-round saccade forward ----------------------------------------

    def _forward_sample(self, sample):
        """Multi-round training with LookHead + optional ClickHead.

        All rounds use LookHead for saccade. When GT is in crop → break.
        After click_phase_step, ClickHead also runs on ALL accumulated
        crop visual tokens for precise positioning.

        Returns (total_loss, num_valid_rounds, round_preds, click_pred, loss_parts).
          click_pred: (x, y) from ClickHead if available, else None.
          loss_parts: diagnostic per-sample decomposition after the same
            normalization as total_loss, with keys total/lm/look/click.
        """
        device = _get_model_device(self.model)
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            zero_parts = {"total": 0.0, "lm": 0.0, "look": 0.0, "click": 0.0}
            return torch.tensor(0.0, device=device, requires_grad=True), 0, [], None, zero_parts

        bbox_gt = sample["bbox_gt"]
        gt_cx = (bbox_gt[0] + bbox_gt[2]) / 2
        gt_cy = (bbox_gt[1] + bbox_gt[3]) / 2

        img_tok = self.model.config.image_token_id
        pp_id = self.model.config.pointer_pad_token_id
        if self.sa.use_dual_tokens:
            look_id = self.model.config.look_pad_token_id
            click_id = self.model.config.click_pad_token_id
            # Use all three as anchor sources (look_pad, click_pad, pointer_pad)
            pp_id = [look_id, click_id, pp_id]
        merge = _get_visual_module(self.model).spatial_merge_size

        self.builder.reset()
        self.model.train()
        total_loss_value = 0.0  # scalar for logging only
        lm_loss_value = 0.0
        look_loss_value = 0.0
        click_loss_value = 0.0
        n_valid = 0
        round_preds = []
        round_responses = []
        click_pred = None  # ClickHead prediction (global coords)

        # Track all crop info for ClickHead at the end
        # Each entry: (vis_embeds_slice, grid_h, grid_w, crop_bbox)
        all_crop_info = []
        crop_hit_round = None  # which round first covered GT (None = never)

        max_rounds = self.max_rounds
        pred_x, pred_y = gt_cx, gt_cy  # overwritten by round 0
        ga_scale = max(self.args.gradient_accumulation_steps, 1)
        look_crop_pixels = self.sa.low_res_max_pixels
        round_crop_bboxes = []

        def _backward_round(round_loss):
            """Backward the accumulated loss for this round, then release graph.
            Gradients accumulate on parameters across rounds."""
            nonlocal total_loss_value
            if round_loss is not None and round_loss.requires_grad:
                scaled = round_loss / (max_rounds * ga_scale)
                scaled.backward()
                total_loss_value += round_loss.item()

        def _attn_np(attn: torch.Tensor):
            return attn.detach().cpu().float().numpy()

        def _fallback_bbox_from_point(center_x, center_y):
            _, fallback_bbox = crop_image(
                img,
                center_x,
                center_y,
                crop_size=self.sa.crop_size,
                crop_upscale=self.sa.crop_upscale,
            )
            return fallback_bbox

        def _proposal_bbox_from_attn(
            img_attn_2d: torch.Tensor,
            *,
            parent_bbox=(0.0, 0.0, 1.0, 1.0),
            fallback_center=None,
        ):
            try:
                proposal_bbox, _, _ = select_top_proposal_bbox(
                    _attn_np(img_attn_2d),
                    parent_bbox=parent_bbox,
                )
            except Exception:
                if fallback_center is None:
                    raise
                proposal_bbox = _fallback_bbox_from_point(fallback_center[0], fallback_center[1])
            _, proposal_bbox = crop_image_bbox(
                img,
                proposal_bbox,
                target_pixels=look_crop_pixels,
            )
            return proposal_bbox

        for ri in range(max_rounds):
            if ri == 0:
                # ===== Round 0: Low-res full image =====
                r_inputs, cur_text, cur_images = self.builder.build_round0(
                    sample, sample["instruction"],
                    use_dual_tokens=self.sa.use_dual_tokens,
                    free_reasoning=self.sa.free_reasoning_sft,
                    sample_sft_reasoning=self.sa.free_reasoning_sft,
                    append_assistant_eos=self.sa.append_assistant_eos,
                )
                if self.sa.free_reasoning_sft:
                    round_responses.append(self.builder.last_assistant_content)
                inp = {k: v.to(device) for k, v in r_inputs.items()}
                # Enable grad on pixel_values for gradient checkpointing
                if inp.get("pixel_values") is not None:
                    inp["pixel_values"] = inp["pixel_values"].requires_grad_(True)

                # LM labels
                lm_labels = None
                lm_weights = None
                if self.sa.lm_loss_weight > 0:
                    lm_labels, lm_weights = make_lm_labels_and_weights(
                        inp["input_ids"],
                        self.tokenizer,
                        round_idx=ri,
                        reasoning_weight=self.sa.lm_reasoning_token_weight,
                        format_weight=self.sa.lm_format_token_weight,
                        look_weight=self.sa.lm_look_token_weight,
                        click_weight=self.sa.lm_click_token_weight,
                    )
                    lm_labels = lm_labels.to(device)
                    lm_weights = lm_weights.to(device)

                outputs = self.model(
                    input_ids=inp["input_ids"],
                    attention_mask=inp.get("attention_mask"),
                    pixel_values=inp["pixel_values"],
                    image_grid_thw=inp.get("image_grid_thw"),
                )
                last_hs = outputs.hidden_states[-1]

                # Accumulate round loss, backward at end of round
                round_loss = torch.tensor(0.0, device=device, requires_grad=True)
                if self.sa.lm_loss_weight > 0 and lm_labels is not None and lm_weights is not None:
                    lm_loss = compute_weighted_lm_loss(outputs.logits, lm_labels, lm_weights)
                    if lm_loss is not None:
                        round_loss = round_loss + self.sa.lm_loss_weight * lm_loss
                        lm_loss_value += self.sa.lm_loss_weight * lm_loss.item()

                # Visual embeds (pre-LLM)
                vis_embeds = self.model.extract_visual_embeds(
                    inp["input_ids"], inp["pixel_values"], inp.get("image_grid_thw"))
                _, vis_ranges = extract_visual_hidden_states(
                    last_hs, inp["input_ids"], img_tok)
                anchor = extract_anchor_hidden_states(
                    last_hs, inp["input_ids"], pp_id, n=0)

                if vis_embeds is None or anchor is None:
                    break

                grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)
                nh0, nw0 = grid_dims[0]

                # LookHead: binary labels on GT region
                look_labels = compute_binary_labels(nh0, nw0, bbox_gt,
                                                    soft=False).to(device).unsqueeze(0)
                attn, look_loss, _ = self.model.dual_head.look(
                    vis_embeds, anchor, labels=look_labels)

                if look_loss is not None:
                    round_loss = round_loss + self.sa.look_loss_weight * look_loss
                    look_loss_value += self.sa.look_loss_weight * look_loss.item()
                    n_valid += 1

                # Backward round 0 loss — releases computation graph
                _backward_round(round_loss)

                with torch.no_grad():
                    # Decode prediction from LookHead attention
                    _, local_idx = identify_attended_image(attn.squeeze(0), vis_ranges)
                    low_attn = attn.squeeze(0)[vis_ranges[0][0]:vis_ranges[0][0]+vis_ranges[0][1]]
                    low_attn_2d = low_attn.view(nh0, nw0)
                    peak_x, peak_y = token_to_spatial(local_idx, nw0, nh0, attn_weights=low_attn)
                    proposal_bbox = _proposal_bbox_from_attn(
                        low_attn_2d,
                        parent_bbox=(0.0, 0.0, 1.0, 1.0),
                        fallback_center=(peak_x, peak_y),
                    )
                    pred_x, pred_y = box_center(proposal_bbox)
                    round_preds.append((pred_x, pred_y))
                    round_crop_bboxes.append(proposal_bbox)

                    # Virtual crop_hit for round 0
                    if point_in_bbox(gt_cx, gt_cy, proposal_bbox):
                        crop_hit_round = 1  # round 0's crop = crop round 1

            else:
                # ===== Round ri: Crop =====
                current_request_bbox = round_crop_bboxes[ri - 1]
                cropped, crop_bbox = crop_image_bbox(
                    img,
                    current_request_bbox,
                    target_pixels=look_crop_pixels,
                )
                gt_in_crop = point_in_bbox(gt_cx, gt_cy, crop_bbox)
                if gt_in_crop and crop_hit_round is None:
                    crop_hit_round = ri  # first round that covered GT

                # Re-build full context: low-res + all crops
                self.builder.reset()
                r_inputs, cur_text, cur_images = self.builder.build_round0(
                    sample, sample["instruction"],
                    use_dual_tokens=self.sa.use_dual_tokens,
                    free_reasoning=self.sa.free_reasoning_sft,
                    assistant_response=(round_responses[0] if self.sa.free_reasoning_sft and round_responses else None),
                    sample_sft_reasoning=(self.sa.free_reasoning_sft and not round_responses),
                    append_assistant_eos=self.sa.append_assistant_eos,
                )
                for prev_ri in range(1, ri):
                    prev_crop_bbox = round_crop_bboxes[prev_ri - 1]
                    prev_crop, prev_bbox = crop_image_bbox(
                        img,
                        prev_crop_bbox,
                        target_pixels=look_crop_pixels,
                    )
                    # Previous crops were always "look" (not the final click)
                    prev_gt_in = point_in_bbox(gt_cx, gt_cy, prev_bbox)
                    try:
                        r_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                            cur_text, cur_images, prev_crop, prev_bbox,
                            gt_in_crop=prev_gt_in, use_dual_tokens=self.sa.use_dual_tokens,
                            free_reasoning=self.sa.free_reasoning_sft,
                            assistant_response=(
                                round_responses[prev_ri]
                                if self.sa.free_reasoning_sft and prev_ri < len(round_responses)
                                else None
                            ),
                            sample_sft_reasoning=(
                                self.sa.free_reasoning_sft and prev_ri >= len(round_responses)
                            ),
                            append_assistant_eos=self.sa.append_assistant_eos,
                        )
                    except Exception:
                        break
                try:
                    r_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                        cur_text, cur_images, cropped, crop_bbox,
                        gt_in_crop=gt_in_crop, use_dual_tokens=self.sa.use_dual_tokens,
                        free_reasoning=self.sa.free_reasoning_sft,
                        sample_sft_reasoning=self.sa.free_reasoning_sft,
                        append_assistant_eos=self.sa.append_assistant_eos,
                    )
                    if self.sa.free_reasoning_sft:
                        if ri < len(round_responses):
                            round_responses[ri] = self.builder.last_assistant_content
                        else:
                            round_responses.append(self.builder.last_assistant_content)
                except Exception:
                    break

                inp = {k: v.to(device) for k, v in r_inputs.items()}
                if inp.get("pixel_values") is not None:
                    inp["pixel_values"] = inp["pixel_values"].requires_grad_(True)

                # LM labels
                lm_labels = None
                lm_weights = None
                if self.sa.lm_loss_weight > 0:
                    lm_labels, lm_weights = make_lm_labels_and_weights(
                        inp["input_ids"],
                        self.tokenizer,
                        round_idx=ri,
                        reasoning_weight=self.sa.lm_reasoning_token_weight,
                        format_weight=self.sa.lm_format_token_weight,
                        look_weight=self.sa.lm_look_token_weight,
                        click_weight=self.sa.lm_click_token_weight,
                    )
                    lm_labels = lm_labels.to(device)
                    lm_weights = lm_weights.to(device)

                outputs = self.model(
                    input_ids=inp["input_ids"],
                    attention_mask=inp.get("attention_mask"),
                    pixel_values=inp["pixel_values"],
                    image_grid_thw=inp.get("image_grid_thw"),
                )
                last_hs = outputs.hidden_states[-1]

                # Accumulate round loss for this crop round
                round_loss = torch.tensor(0.0, device=device, requires_grad=True)
                if self.sa.lm_loss_weight > 0 and lm_labels is not None and lm_weights is not None:
                    lm_loss = compute_weighted_lm_loss(outputs.logits, lm_labels, lm_weights)
                    if lm_loss is not None:
                        round_loss = round_loss + self.sa.lm_loss_weight * lm_loss
                        lm_loss_value += self.sa.lm_loss_weight * lm_loss.item()

                vis_embeds = self.model.extract_visual_embeds(
                    inp["input_ids"], inp["pixel_values"], inp.get("image_grid_thw"))
                _, vis_ranges = extract_visual_hidden_states(
                    last_hs, inp["input_ids"], img_tok)
                anchor = extract_anchor_hidden_states(
                    last_hs, inp["input_ids"], pp_id, n=ri)

                if vis_embeds is None or anchor is None or len(vis_ranges) < ri + 1:
                    _backward_round(round_loss)
                    break

                grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)
                n_low = vis_ranges[0][1]
                n_total = vis_embeds.shape[0]
                latest_img_idx = len(vis_ranges) - 1

                # Mask: current crop overlap in low-res + old crop tokens
                this_crop_mask = compute_overlap_mask(nh0, nw0, crop_bbox)
                full_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
                full_mask[:n_low] = this_crop_mask.to(device)
                for prev_i in range(1, latest_img_idx):
                    off_prev, ntok_prev = vis_ranges[prev_i]
                    full_mask[off_prev:off_prev + ntok_prev] = True

                # Save crop info for ClickHead later
                offset_hi, n_hi = vis_ranges[latest_img_idx]
                nh_high, nw_high = grid_dims[latest_img_idx]
                all_crop_info.append({
                    "offset": offset_hi, "n_tokens": n_hi,
                    "grid_h": nh_high, "grid_w": nw_high,
                    "crop_bbox": crop_bbox, "gt_in_crop": gt_in_crop,
                })

                if gt_in_crop:
                    # GT in crop → LookHead labels on crop patch
                    cbx1, cby1, cbx2, cby2 = crop_bbox
                    cbw, cbh = cbx2 - cbx1, cby2 - cby1
                    if cbw > 0 and cbh > 0:
                        local_gt = (
                            max(0.0, min(1.0, (bbox_gt[0] - cbx1) / cbw)),
                            max(0.0, min(1.0, (bbox_gt[1] - cby1) / cbh)),
                            max(0.0, min(1.0, (bbox_gt[2] - cbx1) / cbw)),
                            max(0.0, min(1.0, (bbox_gt[3] - cby1) / cbh)),
                        )
                        high_labels = compute_binary_labels(nh_high, nw_high, local_gt, soft=False)
                    else:
                        high_labels = torch.zeros(n_hi)

                    # LookHead: labels on crop tokens (coarse: which crop has GT)
                    look_full_labels = torch.zeros(1, n_total, device=device)
                    look_full_labels[0, offset_hi:offset_hi + n_hi] = high_labels.to(device)

                    attn, look_loss, _ = self.model.dual_head.look(
                        vis_embeds, anchor, labels=look_full_labels, mask=full_mask)
                    if look_loss is not None:
                        round_loss = round_loss + self.sa.look_loss_weight * look_loss
                        look_loss_value += self.sa.look_loss_weight * look_loss.item()
                        n_valid += 1

                    # ClickHead (after click_phase_step): precise position on ALL crop tokens
                    if self.in_click_phase and all_crop_info:
                        # Concatenate all crop visual tokens
                        crop_vis_list = []
                        click_label_list = []
                        crop_global_bboxes = []  # for mapping back to global coords
                        for ci in all_crop_info:
                            ci_vis = vis_embeds[ci["offset"]:ci["offset"] + ci["n_tokens"]]
                            crop_vis_list.append(ci_vis)
                            crop_global_bboxes.append(ci["crop_bbox"])

                            if ci["gt_in_crop"]:
                                cbx1, cby1, cbx2, cby2 = ci["crop_bbox"]
                                cbw, cbh = cbx2 - cbx1, cby2 - cby1
                                if cbw > 0 and cbh > 0:
                                    local_gt_ci = (
                                        max(0.0, min(1.0, (bbox_gt[0] - cbx1) / cbw)),
                                        max(0.0, min(1.0, (bbox_gt[1] - cby1) / cbh)),
                                        max(0.0, min(1.0, (bbox_gt[2] - cbx1) / cbw)),
                                        max(0.0, min(1.0, (bbox_gt[3] - cby1) / cbh)),
                                    )
                                    ci_labels = compute_binary_labels(
                                        ci["grid_h"], ci["grid_w"], local_gt_ci, soft=False)
                                else:
                                    ci_labels = torch.zeros(ci["n_tokens"])
                            else:
                                ci_labels = torch.zeros(ci["n_tokens"])
                            click_label_list.append(ci_labels)

                        combined_crop_vis = torch.cat(crop_vis_list, dim=0)  # (sum_n_crop_tokens, d)
                        combined_click_labels = torch.cat(click_label_list, dim=0).to(device).unsqueeze(0)

                        click_attn, click_loss, _ = self.model.dual_head.click(
                            combined_crop_vis, anchor, labels=combined_click_labels)
                        if click_loss is not None:
                            round_loss = round_loss + self.sa.click_loss_weight * click_loss
                            click_loss_value += self.sa.click_loss_weight * click_loss.item()

                        # Decode ClickHead prediction → global coords
                        with torch.no_grad():
                            click_1d = click_attn.squeeze(0)
                            global_argmax = click_1d.argmax().item()
                            # Find which crop this token belongs to
                            running = 0
                            for ci_idx, ci in enumerate(all_crop_info):
                                if running + ci["n_tokens"] > global_argmax:
                                    local_tok = global_argmax - running
                                    ci_nw, ci_nh = ci["grid_w"], ci["grid_h"]
                                    lx = ((local_tok % ci_nw) + 0.5) / ci_nw
                                    ly = ((local_tok // ci_nw) + 0.5) / ci_nh
                                    bx1, by1, bx2, by2 = ci["crop_bbox"]
                                    click_pred = (
                                        bx1 + lx * (bx2 - bx1),
                                        by1 + ly * (by2 - by1),
                                    )
                                    break
                                running += ci["n_tokens"]

                    # Backward round loss before break
                    _backward_round(round_loss)

                    # Record LookHead prediction and break
                    with torch.no_grad():
                        img_idx, local_idx = identify_attended_image(
                            attn.squeeze(0), vis_ranges)
                        if img_idx < len(grid_dims):
                            nh_a, nw_a = grid_dims[img_idx]
                            off_a, n_a = vis_ranges[img_idx]
                            img_attn = attn.squeeze(0)[off_a:off_a+n_a]
                            lx, ly = token_to_spatial(local_idx, nw_a, nh_a, attn_weights=img_attn)
                            info = self.builder.image_infos[img_idx]
                            bx1, by1, bx2, by2 = info.global_bbox
                            round_preds.append((
                                bx1 + lx * (bx2 - bx1),
                                by1 + ly * (by2 - by1),
                            ))
                    break

                else:
                    # GT not in crop → saccade, LookHead labels on low-res
                    low_labels = compute_binary_labels(nh0, nw0, bbox_gt, soft=False)
                    low_labels[this_crop_mask] = 0.0

                    if low_labels.sum() > 0:
                        look_full_labels = torch.zeros(1, n_total, device=device)
                        look_full_labels[0, :n_low] = low_labels.to(device)

                        attn, look_loss, _ = self.model.dual_head.look(
                            vis_embeds, anchor, labels=look_full_labels, mask=full_mask)
                        if look_loss is not None:
                            round_loss = round_loss + self.sa.look_loss_weight * look_loss
                            look_loss_value += self.sa.look_loss_weight * look_loss.item()
                            n_valid += 1
                    else:
                        with torch.no_grad():
                            attn, _, _ = self.model.dual_head.look(
                                vis_embeds, anchor, mask=full_mask)

                    # Backward round loss for saccade round
                    _backward_round(round_loss)

                # Record prediction for saccade rounds
                with torch.no_grad():
                    img_idx, local_idx = identify_attended_image(
                        attn.squeeze(0), vis_ranges)
                    if img_idx < len(grid_dims):
                        nh_a, nw_a = grid_dims[img_idx]
                        off_a, n_a = vis_ranges[img_idx]
                        img_attn = attn.squeeze(0)[off_a:off_a+n_a]
                        lx, ly = token_to_spatial(local_idx, nw_a, nh_a, attn_weights=img_attn)
                        info = self.builder.image_infos[img_idx]
                        bx1, by1, bx2, by2 = info.global_bbox
                        fallback_center = (
                            bx1 + lx * (bx2 - bx1),
                            by1 + ly * (by2 - by1),
                        )
                        proposal_bbox = _proposal_bbox_from_attn(
                            img_attn.view(nh_a, nw_a),
                            parent_bbox=info.global_bbox,
                            fallback_center=fallback_center,
                        )
                        pred_x, pred_y = box_center(proposal_bbox)
                        round_preds.append((pred_x, pred_y))
                        round_crop_bboxes.append(proposal_bbox)
                    else:
                        break

        # crop_rounds: how many rounds to first cover GT (lower = better)
        if crop_hit_round is not None:
            self.metrics["crop_rounds"].append(crop_hit_round)
        else:
            self.metrics["crop_rounds"].append(max_rounds)

        # Track total visual tokens
        try:
            n_img_tokens = (inp["input_ids"][0] == img_tok).sum().item()
            self.metrics["vis_tokens"].append(n_img_tokens)
        except Exception:
            pass

        # Loss was already backward'd each round; return a sample-level
        # decomposition with a shared denominator so total = lm + look + click.
        denom = max(n_valid, 1)
        loss_parts = {
            "total": (lm_loss_value + look_loss_value + click_loss_value) / denom,
            "lm": lm_loss_value / denom,
            "look": look_loss_value / denom,
            "click": click_loss_value / denom,
        }
        return loss_parts["total"], n_valid, round_preds, click_pred, loss_parts

    # -- train step -----------------------------------------------------------

    def _train_step(self, sample):
        loss, nv, round_preds, click_pred, loss_parts = self._forward_sample(sample)
        # Backward already called per-round inside _forward_sample

        gcx = (sample["bbox_gt"][0] + sample["bbox_gt"][2]) / 2
        gcy = (sample["bbox_gt"][1] + sample["bbox_gt"][3]) / 2
        self.metrics["avg_rounds"].append(len(round_preds))

        # hit only from ClickHead (no fallback to LookHead)
        if click_pred is not None:
            final_px, final_py = click_pred
        else:
            final_px, final_py = None, None

        if final_px is not None:
            d = math.sqrt((final_px - gcx)**2 + (final_py - gcy)**2)
            self.metrics["avg_dist"].append(d)
            hit = (sample["bbox_gt"][0] <= final_px <= sample["bbox_gt"][2]
                   and sample["bbox_gt"][1] <= final_py <= sample["bbox_gt"][3])
            self.metrics["hit_rate"].append(1 if hit else 0)
        if nv > 0:
            self.metrics["total_loss"].append(loss_parts["total"])
            self.metrics["lm_loss"].append(loss_parts["lm"])
            self.metrics["look_loss"].append(loss_parts["look"])
            self.metrics["click_loss"].append(loss_parts["click"])
        return loss if nv > 0 else 0.0, nv

    def train_step(self, batch):
        acc_loss = 0.0
        acc_n = 0
        for sample in batch:
            try:
                loss_val, nv = self._train_step(sample)
                if nv > 0:
                    acc_loss += loss_val
                    acc_n += 1
            except Exception:
                traceback.print_exc()
        return acc_loss / max(acc_n, 1), acc_n

    # -- main train loop ------------------------------------------------------

    def train(self):
        bs = self.args.per_device_train_batch_size
        ga = max(self.args.gradient_accumulation_steps, 1)
        epochs = int(self.args.num_train_epochs)
        max_steps = getattr(self.args, 'max_steps', -1) or -1

        if self.rank == 0:
            mode = "Full-param" if not self.sa.use_lora else "LoRA"
            print(f"=== Saccade Foveation Training (Dual Head: LookHead + ClickHead, {mode}) ===")
            print(f"  samples={len(self.train_data)}  epochs={epochs}  bs={bs}  ga={ga}")
            print(f"  low_res={self.sa.low_res_max_pixels}  high_res={self.sa.high_res_max_pixels}")
            print(f"  look_crop_strategy=top1_basin_proposal -> resize_to={self.sa.low_res_max_pixels} pixels")
            print(f"  fallback_square_crop={self.sa.crop_size}x{self.sa.crop_size} x {self.sa.crop_upscale} "
                  f"-> {self.sa.crop_size*self.sa.crop_upscale}x{self.sa.crop_size*self.sa.crop_upscale}  "
                  f"max_saccade_rounds={self.sa.max_saccade_rounds}")
            print(f"  click_phase_step={self.sa.click_phase_step} (ClickHead starts at step {self.sa.click_phase_step})")
            print(f"  lm_loss_weight={self.sa.lm_loss_weight}  look_loss_weight={self.sa.look_loss_weight}  "
                  f"click_loss_weight={self.sa.click_loss_weight}")
            print(f"  lm_token_weights: reason={self.sa.lm_reasoning_token_weight} "
                  f"format={self.sa.lm_format_token_weight} "
                  f"look={self.sa.lm_look_token_weight} click={self.sa.lm_click_token_weight}")
            print(f"  head_lr={self.sa.action_head_lr}  backbone_lr={self.sa.lora_lr}")
            if self.sa.use_lora:
                print(f"  lora_r={self.sa.lora_r}  lora_alpha={self.sa.lora_alpha}")
            print(f"  world_size={self.world_size}  max_steps={max_steps}")

        self.optimizer.zero_grad()
        micro = self.global_step * ga

        for epoch in range(epochs):
            epoch_rng = random.Random(42 + epoch)
            epoch_rng.shuffle(self.train_data)
            if self.world_size > 1:
                shard = self.train_data[self.rank::self.world_size]
            else:
                shard = self.train_data

            skip_samples = self.global_step * ga * bs
            if skip_samples > 0 and skip_samples < len(shard):
                if self.rank == 0:
                    print(f"  Resuming: skipping {skip_samples} samples (step {self.global_step})")
                shard = shard[skip_samples:]
            elif skip_samples >= len(shard):
                if self.rank == 0:
                    print(f"  Epoch {epoch+1} already completed, skipping")
                continue

            pbar = tqdm(range(0, len(shard), bs), desc=f"Epoch {epoch+1}/{epochs}",
                        disable=(self.rank != 0))
            for i in pbar:
                loss_val, nv = self.train_step(shard[i:i+bs])
                micro += 1
                if micro % ga == 0:
                    if self.world_size > 1:
                        for p in self.model.parameters():
                            if p.grad is None:
                                p.grad = torch.zeros_like(p)
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    if any(p.grad is not None for p in self.model.parameters()):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.rank == 0 and self.global_step == self.sa.click_phase_step:
                        print(f"  *** Step {self.global_step}: ClickHead activated ***")

                if self.rank == 0:
                    n_recent = max(self.args.logging_steps * ga * self.world_size * bs, 1)

                    def ar_recent(k, default=None):
                        vals = self.metrics[k][-n_recent:]
                        if vals:
                            return sum(vals) / len(vals)
                        return default

                    all_lrs = self.scheduler.get_last_lr()
                    lr_val = all_lrs[0]
                    bb_lr_val = all_lrs[1] if len(all_lrs) > 1 else lr_val
                    phase = "Look+Click" if self.in_click_phase else "Look"
                    pbar.set_postfix(
                        step=self.global_step,
                        phase=phase,
                        loss=f"{ar_recent('total_loss', loss_val):.3f}" if (self.metrics["total_loss"] or loss_val is not None) else "-",
                        lm=f"{ar_recent('lm_loss', 0.0):.3f}" if self.metrics["lm_loss"] else "-",
                        look=f"{ar_recent('look_loss', 0.0):.3f}" if self.metrics["look_loss"] else "-",
                        click=f"{ar_recent('click_loss', 0.0):.3f}" if self.metrics["click_loss"] else "-",
                        hit=f"{ar_recent('hit_rate', 0.0):.1%}" if self.metrics["hit_rate"] else "-",
                        rounds=f"{ar_recent('avg_rounds', 0.0):.1f}" if self.metrics["avg_rounds"] else "-",
                        head_lr=f"{lr_val:.2e}",
                        bb_lr=f"{bb_lr_val:.2e}",
                        refresh=False,
                    )

                # Logging
                if self.rank == 0 and self.global_step % self.args.logging_steps == 0:
                    if micro % ga == 0:
                        n_since_last = self.args.logging_steps * ga * self.world_size * bs
                        def ar(k, n=n_since_last):
                            vals = self.metrics[k][-n:]
                            return sum(vals) / max(len(vals), 1)
                        all_lrs = self.scheduler.get_last_lr()
                        lr_val = all_lrs[0]
                        bb_lr_val = all_lrs[1] if len(all_lrs) > 1 else lr_val
                        loss_str = (
                            f" loss={ar('total_loss'):.4f}"
                            if self.metrics["total_loss"]
                            else f" loss={loss_val:.4f}"
                        )
                        lm_str = f" lm={ar('lm_loss'):.3f}" if self.metrics["lm_loss"] else ""
                        look_str = f" look={ar('look_loss'):.3f}" if self.metrics["look_loss"] else ""
                        click_str = f" click={ar('click_loss'):.3f}" if self.metrics["click_loss"] else ""
                        phase = "Look+Click" if self.in_click_phase else "Look"
                        print(f"  [Step {self.global_step}] [{phase}]{loss_str}{lm_str}{look_str}{click_str} "
                              f"hit={ar('hit_rate'):.1%} dist={ar('avg_dist'):.4f} "
                              f"rounds={ar('avg_rounds'):.1f} "
                              f"crop_rounds={ar('crop_rounds'):.1f} vis_tok={ar('vis_tokens'):.0f} "
                              f"head_lr={lr_val:.2e} bb_lr={bb_lr_val:.2e}")
                        for key in list(self.metrics.keys()):
                            if len(self.metrics[key]) > 500:
                                self.metrics[key] = self.metrics[key][-500:]

                if self.global_step % self.args.save_steps == 0 and self.global_step > 0:
                    self._save(f"checkpoint-{self.global_step}")

                if max_steps > 0 and self.global_step >= max_steps:
                    if self.rank == 0:
                        print(f"Reached max_steps={max_steps}, stopping.")
                    self._save(f"checkpoint-{self.global_step}")
                    return

        self._save("final")
        if self.rank == 0:
            print(f"Done. Saved to {self.args.output_dir}")

    def _save(self, name):
        if self.world_size > 1:
            dist.barrier()
        p = os.path.join(self.args.output_dir, name)

        if self.rank != 0:
            return
        os.makedirs(p, exist_ok=True)
        self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }, os.path.join(p, "training_state.pt"))

        if self.rank == 0:
            with open(os.path.join(p, "metrics.json"), "w") as f:
                json.dump({k: v[-100:] for k, v in self.metrics.items()}, f)
            print(f"  Saved: {p}")


# -- Main --------------------------------------------------------------------

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank() if dist.is_initialized() else 0

    parser = transformers.HfArgumentParser((ScriptArgs, TrainArgs))
    sa, ta = parser.parse_args_into_dataclasses()

    if rank == 0:
        print(f"Building model: {sa.model_name_or_path}")
    model, tokenizer, processor = build_model(
        sa.model_name_or_path,
        lora_r=sa.lora_r,
        lora_alpha=sa.lora_alpha,
        lora_target_modules=sa.lora_target_modules,
        torch_dtype=torch.bfloat16 if ta.bf16 else None,
        gradient_checkpointing=ta.gradient_checkpointing,
        use_lora=sa.use_lora,
        click_head_from=sa.click_head_from,
    )

    # Load checkpoint weights for init/resume
    ckpt_to_load = sa.resume_ckpt or sa.init_ckpt
    if ckpt_to_load:
        ckpt = ckpt_to_load
        if rank == 0:
            mode = "resume" if sa.resume_ckpt else "init-only"
            print(f"Loading model weights from checkpoint ({mode}): {ckpt}")
        if sa.use_lora:
            from peft import set_peft_model_state_dict
            adapter_file = os.path.join(ckpt, "adapter_model.safetensors")
            if os.path.exists(adapter_file):
                from safetensors.torch import load_file
                adapter_state = load_file(adapter_file)
            else:
                adapter_state = torch.load(os.path.join(ckpt, "adapter_model.bin"),
                                           map_location="cpu", weights_only=True)
            set_peft_model_state_dict(model.backbone, adapter_state)
        else:
            model_file = os.path.join(ckpt, "model.safetensors")
            if os.path.exists(model_file):
                from safetensors.torch import load_file
                state = load_file(model_file)
                model.backbone.load_state_dict(state, strict=False)

        # Load dual head (or old action_head with backward compat)
        dual_path = os.path.join(ckpt, "dual_head.pt")
        old_path = os.path.join(ckpt, "action_head.pt")
        if os.path.exists(dual_path):
            model.dual_head.load_state_dict(
                torch.load(dual_path, map_location="cpu", weights_only=True))
        elif os.path.exists(old_path):
            old_state = torch.load(old_path, map_location="cpu", weights_only=True)
            look_state = {}
            for k, v in old_state.items():
                if k.startswith("bbox_head") or k == "beta":
                    continue
                look_state[f"look_head.{k}"] = v
            model.dual_head.load_state_dict(look_state, strict=False)
            if rank == 0:
                print(f"  Loaded old ActionHead → LookHead (backward compat)")
        if rank == 0:
            print(f"  Model weights loaded from {ckpt}")

    model.to(f"cuda:{local_rank}")

    train_data = load_dataset(sa.data_path, sa.image_folder, sa.max_samples, sa.max_samples_per_dataset)
    os.makedirs(ta.output_dir, exist_ok=True)

    trainer = SaccadeTrainer(model, tokenizer, train_data, ta, sa)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
