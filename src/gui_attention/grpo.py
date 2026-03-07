"""GRPO (Group Relative Policy Optimization) for Saccade Foveation.

Hybrid training: GRPO for saccade strategy + SFT for head precision.

For each sample, generate G trajectories by sampling from attention distributions.
Reward = hit (click in GT bbox) - α * rounds_used.
GRPO updates the model to favor high-reward trajectories.

Key insight: LookHead and ClickHead output attention distributions (softmax logits).
Instead of argmax, we sample from these distributions to get diverse trajectories.
The log-probability of each sampled action enables policy gradient updates.

Usage:
    torchrun --nproc_per_node=8 src/gui_attention/grpo.py \
        --model_name_or_path /path/to/sft_checkpoint \
        --data_path /path/to/data.json \
        --image_folder /path/to/images \
        --output_dir /path/to/output
"""

import os
import random
import traceback
import math
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import StoppingCriteriaList

from gui_attention.attention import (
    extract_anchor_hidden_states,
    extract_visual_hidden_states,
)
from gui_attention.builder import MultiRoundInputBuilder
from gui_attention.constants import (
    HIGH_RES_MAX_PIXELS,
    LOW_RES_MAX_PIXELS,
)
from gui_attention.crop import crop_image, point_in_bbox
from gui_attention.labels import compute_overlap_mask
from gui_attention.model import Qwen25VLWithDualHead, build_model
from gui_attention.reasoning import (
    ActionSpanStoppingCriteria,
    decode_reasoning_content,
    parse_reasoning_action,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -- Arguments ----------------------------------------------------------------

@dataclass
class GRPOArgs:
    model_name_or_path: str = field(default="microsoft/GUI-Actor-3B-Qwen2.5-VL")
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    max_samples: Optional[int] = field(default=None)
    min_pixels: int = field(default=3136)
    low_res_max_pixels: int = field(default=LOW_RES_MAX_PIXELS)
    high_res_max_pixels: int = field(default=HIGH_RES_MAX_PIXELS)
    crop_size: int = field(default=308)
    crop_upscale: int = field(default=3)
    max_saccade_rounds: int = field(default=6)
    reasoning_max_new_tokens: int = field(default=48)
    use_dual_tokens: bool = field(default=True)
    # GRPO
    group_size: int = field(default=8, metadata={"help": "Number of trajectories per sample (G)"})
    reward_hit: float = field(default=1.0, metadata={"help": "Reward for clicking in GT bbox"})
    reward_proximity_weight: float = field(default=0.25, metadata={"help": "Dense reward weight for clicking closer to the GT center."})
    reward_round_penalty: float = field(default=0.02, metadata={"help": "Penalty per round used"})
    reward_overlap_penalty: float = field(default=0.1, metadata={"help": "Penalty weight for crop overlap (IoU with previous crops)"})
    reward_format: float = field(default=0.05, metadata={"help": "Reward for producing a valid reasoning/action format."})
    reward_malformed_penalty: float = field(default=0.05, metadata={"help": "Penalty for malformed reasoning/action outputs."})
    kl_coeff: float = field(default=0.01, metadata={"help": "KL penalty coefficient against reference"})
    clip_eps: float = field(default=0.2, metadata={"help": "PPO-style clipping epsilon"})
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature for attention distributions"})
    # Loss weights (SFT components still used as auxiliary)
    look_sft_weight: float = field(default=0.1, metadata={"help": "Weight for LookHead SFT KL loss (auxiliary)"})
    click_sft_weight: float = field(default=0.1, metadata={"help": "Weight for ClickHead SFT KL loss (auxiliary)"})
    lm_loss_weight: float = field(default=0.1)
    # LoRA
    use_lora: bool = field(default=True)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_target_modules: str = field(default="q_proj,v_proj")
    # LR
    action_head_lr: float = field(default=1e-5, metadata={"help": "LR for heads (lower for GRPO fine-tuning)"})
    lora_lr: float = field(default=5e-6, metadata={"help": "LR for backbone (lower for GRPO fine-tuning)"})
    # Data
    max_samples_per_dataset: Optional[str] = field(default=None)
    # Init
    click_head_from: Optional[str] = field(default=None)
    resume_ckpt: Optional[str] = field(default=None)


@dataclass
class GRPOTrainArgs(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    gradient_checkpointing: bool = field(default=True)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)


# -- Helpers ------------------------------------------------------------------

def _get_model_device(model):
    return next(model.parameters()).device


def _get_visual_module(model):
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


# -- Trajectory ---------------------------------------------------------------

@dataclass
class Trajectory:
    """A single saccade trajectory for one sample."""
    round_preds: list                   # [(x, y)|None, ...] per round; look rounds carry next focus
    round_crop_bboxes: list             # [None|bbox, ...] aligned with rounds
    look_token_indices: list            # [None|idx, ...] aligned with rounds
    round_actions: list                 # ["look"|"click", ...] aligned with rounds
    round_generated_token_ids: list     # [list[int], ...] LM-generated reasoning continuation tokens
    round_raw_responses: list           # raw generated assistant content per round
    round_used_responses: list          # repaired/canonical assistant content used for replay
    round_format_oks: list              # whether each round followed the requested format
    round_parse_failures: list          # whether each round required fallback repair
    round_reason_tokens: list           # generated token count per round
    click_token_index: Optional[int]    # sampled ClickHead token idx on concatenated crop tokens
    click_pred: tuple                   # (x, y) final click position
    n_rounds: int                       # total rounds used
    hit: bool                           # whether click was in GT bbox
    proximity_score: float              # distance-based click quality score in [0, 1]
    format_ok_rate: float               # fraction of rounds with valid format
    parse_fail_rate: float              # fraction of rounds that needed repair
    avg_reason_tokens: float            # average number of generated reasoning tokens
    raw_reward: float                   # pre-normalization reward
    reward: float                       # normalized reward / advantage


# -- GRPO Trainer -------------------------------------------------------------

class GRPOTrainer:
    def __init__(self, model: Qwen25VLWithDualHead, ref_dual_head,
                 tokenizer, train_data, args: GRPOTrainArgs, grpo_args: GRPOArgs):
        self.model = model
        self.ref_dual_head = ref_dual_head  # frozen CPU copy for KL on dual head only
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.args = args
        self.ga = grpo_args
        self.builder = MultiRoundInputBuilder(
            grpo_args.model_name_or_path, tokenizer, grpo_args.min_pixels,
            low_res_max_pixels=grpo_args.low_res_max_pixels,
            high_res_max_pixels=grpo_args.high_res_max_pixels,
        )

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        total_steps = (
            int(args.num_train_epochs) * len(train_data)
            // self.world_size
            // max(args.per_device_train_batch_size, 1)
            // max(args.gradient_accumulation_steps, 1)
        )

        head_params = list(model.dual_head.parameters())
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        param_groups = [
            {"params": head_params, "lr": grpo_args.action_head_lr},
            {"params": backbone_params, "lr": grpo_args.lora_lr},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        warmup_steps = int(total_steps * args.warmup_ratio) if args.warmup_ratio > 0 else 0
        self.scheduler = transformers.get_cosine_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=max(total_steps, 1),
        )

        self.metrics = defaultdict(list)
        self.global_step = 0

    def _get_pointer_pad_ids(self):
        pp_id = self.model.config.pointer_pad_token_id
        if self.ga.use_dual_tokens and hasattr(self.model.config, "look_pad_token_id") and hasattr(self.model.config, "click_pad_token_id"):
            pp_id = [
                self.model.config.look_pad_token_id,
                self.model.config.click_pad_token_id,
                self.model.config.pointer_pad_token_id,
            ]
        return pp_id

    @staticmethod
    def _locate_visual_token(token_idx: int, visual_ranges):
        for img_idx, (offset, n_tokens) in enumerate(visual_ranges):
            if offset <= token_idx < offset + n_tokens:
                return img_idx, token_idx - offset
        return None, None

    def _continuation_log_prob(self, inp, continuation_ids) -> torch.Tensor:
        """Log-probability of an autoregressive continuation under the current prompt."""
        device = inp["input_ids"].device
        if not continuation_ids:
            return torch.tensor(0.0, device=device)
        cont = torch.tensor(continuation_ids, dtype=torch.long, device=device)

        full_input_ids = torch.cat([inp["input_ids"], cont.unsqueeze(0)], dim=1)
        attention_mask = inp.get("attention_mask")
        if attention_mask is not None:
            suffix_mask = torch.ones(
                (attention_mask.shape[0], cont.numel()),
                dtype=attention_mask.dtype,
                device=device,
            )
            attention_mask = torch.cat([attention_mask, suffix_mask], dim=1)

        outputs = self.model(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            pixel_values=inp.get("pixel_values"),
            image_grid_thw=inp.get("image_grid_thw"),
        )
        prompt_len = inp["input_ids"].shape[1]
        logits = outputs.logits[:, prompt_len - 1:-1, :]
        token_log_probs = F.log_softmax(logits, dim=-1).gather(
            -1, cont.view(1, -1, 1)
        ).squeeze(-1)
        return token_log_probs.sum()

    def _build_reasoning_context(
        self,
        sample,
        img,
        history_used_responses,
        history_points,
        *,
        current_crop=None,
        current_bbox=None,
        current_response=None,
        for_generation=False,
    ):
        self.builder.reset()
        if history_used_responses:
            r_inputs, cur_text, cur_images = self.builder.build_round0(
                sample,
                sample["instruction"],
                use_dual_tokens=self.ga.use_dual_tokens,
                free_reasoning=True,
                assistant_response=history_used_responses[0],
            )
        else:
            r_inputs, cur_text, cur_images = self.builder.build_round0(
                sample,
                sample["instruction"],
                use_dual_tokens=self.ga.use_dual_tokens,
                free_reasoning=True,
                for_generation=(for_generation and current_crop is None),
                assistant_response=(current_response if current_crop is None else None),
            )

        for hist_idx in range(1, len(history_used_responses)):
            prev_px, prev_py = history_points[hist_idx - 1]
            prev_crop, prev_bbox = crop_image(
                img,
                prev_px,
                prev_py,
                crop_size=self.ga.crop_size,
                crop_upscale=self.ga.crop_upscale,
            )
            r_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                cur_text,
                cur_images,
                prev_crop,
                prev_bbox,
                use_dual_tokens=self.ga.use_dual_tokens,
                free_reasoning=True,
                assistant_response=history_used_responses[hist_idx],
            )

        if current_crop is not None:
            r_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                cur_text,
                cur_images,
                current_crop,
                current_bbox,
                use_dual_tokens=self.ga.use_dual_tokens,
                free_reasoning=True,
                assistant_response=current_response,
                for_generation=for_generation,
            )
        return r_inputs, cur_text, cur_images

    def _generate_reasoning_response(self, inp, *, allow_click: bool, do_sample: bool):
        device = inp["input_ids"].device
        stop = StoppingCriteriaList([
            ActionSpanStoppingCriteria(
                prompt_len=inp["input_ids"].shape[1],
                pointer_end_token_id=self.model.config.pointer_end_token_id,
                allowed_action_token_ids=[
                    getattr(self.model.config, "look_pad_token_id", None),
                    getattr(self.model.config, "click_pad_token_id", self.model.config.pointer_pad_token_id),
                ],
            )
        ])

        gen_kwargs = dict(
            input_ids=inp["input_ids"],
            attention_mask=inp.get("attention_mask"),
            pixel_values=inp.get("pixel_values"),
            image_grid_thw=inp.get("image_grid_thw"),
            max_new_tokens=self.ga.reasoning_max_new_tokens,
            do_sample=do_sample,
            stopping_criteria=stop,
        )
        if do_sample:
            gen_kwargs["temperature"] = max(self.ga.temperature, 1e-5)

        with torch.no_grad():
            gen_ids = self.model.generate(**gen_kwargs)

        new_ids = gen_ids[0, inp["input_ids"].shape[1]:].tolist()
        raw_content = decode_reasoning_content(self.tokenizer, new_ids)
        parsed = parse_reasoning_action(raw_content, allow_click=allow_click)
        return parsed, new_ids

    # -- generate one trajectory (sampling) -----------------------------------

    def _generate_trajectory(self, sample, img) -> Optional[Trajectory]:
        """Run one free-reasoning trajectory with sampled LM reasoning and action heads."""
        device = _get_model_device(self.model)
        bbox_gt = sample["bbox_gt"]
        img_tok = self.model.config.image_token_id
        pp_id = self._get_pointer_pad_ids()
        merge = _get_visual_module(self.model).spatial_merge_size
        temperature = self.ga.temperature

        round_preds = []
        round_crop_bboxes = []
        look_token_indices = []
        round_actions = []
        round_generated_token_ids = []
        round_raw_responses = []
        round_used_responses = []
        round_format_oks = []
        round_parse_failures = []
        round_reason_tokens = []
        all_crop_info = []

        for ri in range(self.ga.max_saccade_rounds):
            if ri == 0:
                try:
                    prompt_inputs, _, _ = self._build_reasoning_context(
                        sample,
                        img,
                        [],
                        [],
                        for_generation=True,
                    )
                except Exception:
                    return None
                inp = {k: v.to(device) for k, v in prompt_inputs.items()}
                parsed, gen_ids = self._generate_reasoning_response(
                    inp,
                    allow_click=False,
                    do_sample=True,
                )

                round_crop_bboxes.append(None)
                round_actions.append(parsed.action)
                round_generated_token_ids.append(gen_ids)
                round_raw_responses.append(parsed.raw_content)
                round_used_responses.append(parsed.used_content)
                round_format_oks.append(parsed.format_ok)
                round_parse_failures.append(parsed.parse_failed)
                round_reason_tokens.append(len(gen_ids))

                try:
                    r_inputs, _, _ = self._build_reasoning_context(
                        sample,
                        img,
                        [],
                        [],
                        current_response=parsed.used_content,
                    )
                except Exception:
                    return None
            else:
                prev_focus = round_preds[-1]
                if prev_focus is None:
                    return None
                cropped, crop_bbox = crop_image(
                    img,
                    prev_focus[0],
                    prev_focus[1],
                    crop_size=self.ga.crop_size,
                    crop_upscale=self.ga.crop_upscale,
                )

                history_responses = list(round_used_responses)
                history_points = list(round_preds)
                try:
                    prompt_inputs, _, _ = self._build_reasoning_context(
                        sample,
                        img,
                        history_responses,
                        history_points,
                        current_crop=cropped,
                        current_bbox=crop_bbox,
                        for_generation=True,
                    )
                except Exception:
                    return None
                inp = {k: v.to(device) for k, v in prompt_inputs.items()}
                parsed, gen_ids = self._generate_reasoning_response(
                    inp,
                    allow_click=True,
                    do_sample=True,
                )

                round_crop_bboxes.append(crop_bbox)
                round_actions.append(parsed.action)
                round_generated_token_ids.append(gen_ids)
                round_raw_responses.append(parsed.raw_content)
                round_used_responses.append(parsed.used_content)
                round_format_oks.append(parsed.format_ok)
                round_parse_failures.append(parsed.parse_failed)
                round_reason_tokens.append(len(gen_ids))

                try:
                    r_inputs, _, _ = self._build_reasoning_context(
                        sample,
                        img,
                        history_responses,
                        history_points,
                        current_crop=cropped,
                        current_bbox=crop_bbox,
                        current_response=parsed.used_content,
                    )
                except Exception:
                    return None

            inp = {k: v.to(device) for k, v in r_inputs.items()}
            outputs = self.model(
                input_ids=inp["input_ids"],
                attention_mask=inp.get("attention_mask"),
                pixel_values=inp.get("pixel_values"),
                image_grid_thw=inp.get("image_grid_thw"),
            )
            last_hs = outputs.hidden_states[-1]

            vis_embeds = self.model.extract_visual_embeds(
                inp["input_ids"], inp.get("pixel_values"), inp.get("image_grid_thw"))
            _, vis_ranges = extract_visual_hidden_states(
                last_hs, inp["input_ids"], img_tok)
            anchor = extract_anchor_hidden_states(
                last_hs, inp["input_ids"], pp_id, n=ri)

            if vis_embeds is None or anchor is None:
                return None

            grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)

            if ri == 0:
                nh0, nw0 = grid_dims[0]
                _, _, logits = self.model.dual_head.look(vis_embeds, anchor)
                logits_1d = logits.squeeze(0) / temperature
                probs = F.softmax(logits_1d, dim=-1)
                sampled_idx = torch.multinomial(probs, 1).item()
                pred_x = ((sampled_idx % nw0) + 0.5) / nw0
                pred_y = ((sampled_idx // nw0) + 0.5) / nh0
                round_preds.append((pred_x, pred_y))
                look_token_indices.append(sampled_idx)
                continue

            n_total = vis_embeds.shape[0]
            n_low = vis_ranges[0][1]
            latest_img_idx = len(vis_ranges) - 1
            this_crop_mask = compute_overlap_mask(nh0, nw0, crop_bbox)
            full_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
            full_mask[:n_low] = this_crop_mask.to(device)
            for prev_i in range(1, latest_img_idx):
                off_prev, ntok_prev = vis_ranges[prev_i]
                full_mask[off_prev:off_prev + ntok_prev] = True

            offset_hi, n_hi = vis_ranges[latest_img_idx]
            nh_high, nw_high = grid_dims[latest_img_idx]
            all_crop_info.append({
                "offset": offset_hi,
                "n_tokens": n_hi,
                "grid_h": nh_high,
                "grid_w": nw_high,
                "crop_bbox": crop_bbox,
            })

            if parsed.action == "click":
                crop_vis_list = [
                    vis_embeds[ci["offset"]:ci["offset"] + ci["n_tokens"]]
                    for ci in all_crop_info
                ]
                combined_crop_vis = torch.cat(crop_vis_list, dim=0)
                _, _, click_logits = self.model.dual_head.click(combined_crop_vis, anchor)
                click_logits_1d = click_logits.squeeze(0) / temperature
                click_probs = F.softmax(click_logits_1d, dim=-1)
                click_sampled = torch.multinomial(click_probs, 1).item()

                running = 0
                click_pred = (0.5, 0.5)
                for ci in all_crop_info:
                    if running + ci["n_tokens"] > click_sampled:
                        local_tok = click_sampled - running
                        lx = ((local_tok % ci["grid_w"]) + 0.5) / ci["grid_w"]
                        ly = ((local_tok // ci["grid_w"]) + 0.5) / ci["grid_h"]
                        bx1, by1, bx2, by2 = ci["crop_bbox"]
                        click_pred = (
                            bx1 + lx * (bx2 - bx1),
                            by1 + ly * (by2 - by1),
                        )
                        break
                    running += ci["n_tokens"]

                round_preds.append(None)
                look_token_indices.append(None)
                hit = point_in_bbox(click_pred[0], click_pred[1], bbox_gt)
                format_ok_rate = sum(1 for ok in round_format_oks if ok) / len(round_format_oks)
                parse_fail_rate = sum(1 for bad in round_parse_failures if bad) / len(round_parse_failures)
                avg_reason_tokens = sum(round_reason_tokens) / len(round_reason_tokens)
                return Trajectory(
                    round_preds=round_preds,
                    round_crop_bboxes=round_crop_bboxes,
                    look_token_indices=look_token_indices,
                    round_actions=round_actions,
                    round_generated_token_ids=round_generated_token_ids,
                    round_raw_responses=round_raw_responses,
                    round_used_responses=round_used_responses,
                    round_format_oks=round_format_oks,
                    round_parse_failures=round_parse_failures,
                    round_reason_tokens=round_reason_tokens,
                    click_token_index=click_sampled,
                    click_pred=click_pred,
                    n_rounds=len(round_actions),
                    hit=hit,
                    proximity_score=0.0,
                    format_ok_rate=format_ok_rate,
                    parse_fail_rate=parse_fail_rate,
                    avg_reason_tokens=avg_reason_tokens,
                    raw_reward=0.0,
                    reward=0.0,
                )

            _, _, logits = self.model.dual_head.look(vis_embeds, anchor, mask=full_mask)
            logits_1d = logits.squeeze(0) / temperature
            probs = F.softmax(logits_1d, dim=-1)
            sampled_idx = torch.multinomial(probs, 1).item()
            img_idx, local_sampled = self._locate_visual_token(sampled_idx, vis_ranges)
            if img_idx is None or img_idx >= len(grid_dims) or img_idx >= len(self.builder.image_infos):
                return None

            nh_a, nw_a = grid_dims[img_idx]
            info = self.builder.image_infos[img_idx]
            bx1, by1, bx2, by2 = info.global_bbox
            pred_x = bx1 + (((local_sampled % nw_a) + 0.5) / nw_a) * (bx2 - bx1)
            pred_y = by1 + (((local_sampled // nw_a) + 0.5) / nh_a) * (by2 - by1)
            round_preds.append((pred_x, pred_y))
            look_token_indices.append(sampled_idx)

        click_pred = next((p for p in reversed(round_preds) if p is not None), (0.5, 0.5))
        hit = point_in_bbox(click_pred[0], click_pred[1], bbox_gt)
        format_ok_rate = sum(1 for ok in round_format_oks if ok) / len(round_format_oks)
        parse_fail_rate = sum(1 for bad in round_parse_failures if bad) / len(round_parse_failures)
        avg_reason_tokens = sum(round_reason_tokens) / len(round_reason_tokens)
        return Trajectory(
            round_preds=round_preds,
            round_crop_bboxes=round_crop_bboxes,
            look_token_indices=look_token_indices,
            round_actions=round_actions,
            round_generated_token_ids=round_generated_token_ids,
            round_raw_responses=round_raw_responses,
            round_used_responses=round_used_responses,
            round_format_oks=round_format_oks,
            round_parse_failures=round_parse_failures,
            round_reason_tokens=round_reason_tokens,
            click_token_index=None,
            click_pred=click_pred,
            n_rounds=len(round_actions),
            hit=hit,
            proximity_score=0.0,
            format_ok_rate=format_ok_rate,
            parse_fail_rate=parse_fail_rate,
            avg_reason_tokens=avg_reason_tokens,
            raw_reward=0.0,
            reward=0.0,
        )

    # -- compute rewards ------------------------------------------------------

    @staticmethod
    def _bbox_iou(a, b):
        """IoU between two (x1, y1, x2, y2) bboxes."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _compute_proximity_score(click_pred, bbox_gt):
        """Continuous click quality based on distance to GT bbox center.

        We only have GT bbox annotations, so center distance is the most stable
        dense proxy for "how good was the click" beyond binary hit/miss.
        """
        bx1, by1, bx2, by2 = bbox_gt
        gt_cx = (bx1 + bx2) / 2
        gt_cy = (by1 + by2) / 2
        bbox_diag = math.hypot(bx2 - bx1, by2 - by1)
        norm = max(bbox_diag, 0.05)
        dist = math.hypot(click_pred[0] - gt_cx, click_pred[1] - gt_cy)
        return math.exp(-0.5 * (dist / norm) ** 2)

    def _compute_overlap_penalty(self, crop_bboxes):
        """Sum of max IoU with any previous crop, for each crop."""
        total = 0.0
        for i, bbox in enumerate(crop_bboxes):
            if i == 0:
                continue
            max_iou = max(self._bbox_iou(bbox, crop_bboxes[j]) for j in range(i))
            total += max_iou
        return total

    def _compute_rewards(self, sample, trajectories: list):
        """Compute rewards for a group of trajectories (same sample).

        reward = hit * reward_hit
                 + proximity_weight * proximity_score
                 + format_reward * format_ok_rate
                 - malformed_penalty * parse_fail_rate
                 - round_penalty * n_rounds
                 - overlap_penalty * sum_of_max_ious
        Then group-normalize (GRPO).
        """
        for traj in trajectories:
            r = 0.0
            proximity = self._compute_proximity_score(traj.click_pred, sample["bbox_gt"])
            traj.proximity_score = proximity
            if traj.hit:
                r += self.ga.reward_hit
            r += self.ga.reward_proximity_weight * proximity
            r += self.ga.reward_format * traj.format_ok_rate
            r -= self.ga.reward_malformed_penalty * traj.parse_fail_rate
            r -= self.ga.reward_round_penalty * traj.n_rounds
            # Overlap penalty: penalize revisiting same areas
            crop_bboxes = [bbox for bbox in traj.round_crop_bboxes if bbox is not None]
            if crop_bboxes and self.ga.reward_overlap_penalty > 0:
                r -= self.ga.reward_overlap_penalty * self._compute_overlap_penalty(crop_bboxes)
            traj.raw_reward = r

        # Group normalization
        rewards = [t.raw_reward for t in trajectories]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        std_r = max(std_r, 1e-8)

        for traj in trajectories:
            traj.reward = (traj.raw_reward - mean_r) / std_r  # advantage

    # -- GRPO loss ------------------------------------------------------------

    def _compute_grpo_loss(self, sample, trajectories):
        """Compute GRPO policy gradient loss.

        loss = -Σ advantage_i * (Σ look_log_probs + click_log_prob)
               + β * KL(π || π_ref)

        Also includes optional SFT auxiliary losses for head precision.
        """
        device = _get_model_device(self.model)

        # Policy gradient: weight log_probs by advantage
        pg_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for traj in trajectories:
            if abs(traj.reward) < 1e-8:
                continue  # zero advantage, skip

            # Replay trajectory to get differentiable log_probs
            traj_loss = self._replay_trajectory_loss(sample, traj)
            if traj_loss is not None:
                pg_loss = pg_loss + traj_loss

        return pg_loss

    def _replay_trajectory_loss(self, sample, traj: Trajectory):
        """Replay a trajectory and compute differentiable policy gradient loss.

        Returns: -advantage * sum(log_probs)
        """
        device = _get_model_device(self.model)
        img_tok = self.model.config.image_token_id
        pp_id = self._get_pointer_pad_ids()
        merge = _get_visual_module(self.model).spatial_merge_size

        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return None

        self.model.train()
        temperature = self.ga.temperature
        total_log_prob = torch.tensor(0.0, device=device)
        all_crop_info = []
        nh0 = nw0 = None

        for ri in range(traj.n_rounds):
            history_responses = traj.round_used_responses[:ri]
            history_points = [p for p in traj.round_preds[:ri] if p is not None]

            if ri == 0:
                try:
                    prompt_inputs, _, _ = self._build_reasoning_context(
                        sample,
                        img,
                        history_responses,
                        history_points,
                        for_generation=True,
                    )
                    r_inputs, _, _ = self._build_reasoning_context(
                        sample,
                        img,
                        history_responses,
                        history_points,
                        current_response=traj.round_used_responses[ri],
                    )
                except Exception:
                    return None
            else:
                prev_focus = traj.round_preds[ri - 1]
                if prev_focus is None:
                    return None
                cropped, _ = crop_image(
                    img,
                    prev_focus[0],
                    prev_focus[1],
                    crop_size=self.ga.crop_size,
                    crop_upscale=self.ga.crop_upscale,
                )
                crop_bbox = traj.round_crop_bboxes[ri]
                if crop_bbox is None:
                    return None
                try:
                    prompt_inputs, _, _ = self._build_reasoning_context(
                        sample,
                        img,
                        history_responses,
                        history_points,
                        current_crop=cropped,
                        current_bbox=crop_bbox,
                        for_generation=True,
                    )
                    r_inputs, _, _ = self._build_reasoning_context(
                        sample,
                        img,
                        history_responses,
                        history_points,
                        current_crop=cropped,
                        current_bbox=crop_bbox,
                        current_response=traj.round_used_responses[ri],
                    )
                except Exception:
                    return None

            prompt_inp = {k: v.to(device) for k, v in prompt_inputs.items()}
            total_log_prob = total_log_prob + self._continuation_log_prob(
                prompt_inp,
                traj.round_generated_token_ids[ri],
            )

            inp = {k: v.to(device) for k, v in r_inputs.items()}
            outputs = self.model(
                input_ids=inp["input_ids"],
                attention_mask=inp.get("attention_mask"),
                pixel_values=inp.get("pixel_values"),
                image_grid_thw=inp.get("image_grid_thw"),
            )
            last_hs = outputs.hidden_states[-1]
            vis_embeds = self.model.extract_visual_embeds(
                inp["input_ids"], inp.get("pixel_values"), inp.get("image_grid_thw"))
            _, vis_ranges = extract_visual_hidden_states(
                last_hs, inp["input_ids"], img_tok)
            anchor = extract_anchor_hidden_states(
                last_hs, inp["input_ids"], pp_id, n=ri)

            if vis_embeds is None or anchor is None:
                return None

            grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)
            if ri == 0:
                nh0, nw0 = grid_dims[0]
                _, _, logits = self.model.dual_head.look(vis_embeds, anchor)
                logits_1d = logits.squeeze(0) / temperature
                sampled_idx = traj.look_token_indices[ri]
                if sampled_idx is None or sampled_idx >= logits_1d.shape[0]:
                    return None
                total_log_prob = total_log_prob + F.log_softmax(logits_1d, dim=-1)[sampled_idx]
                continue

            crop_bbox = traj.round_crop_bboxes[ri]
            if crop_bbox is None or nh0 is None or nw0 is None:
                return None
            n_total = vis_embeds.shape[0]
            n_low = vis_ranges[0][1]
            latest_img_idx = len(vis_ranges) - 1
            this_crop_mask = compute_overlap_mask(nh0, nw0, crop_bbox)
            full_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
            full_mask[:n_low] = this_crop_mask.to(device)
            for prev_i in range(1, latest_img_idx):
                off_prev, ntok_prev = vis_ranges[prev_i]
                full_mask[off_prev:off_prev + ntok_prev] = True

            offset_hi, n_hi = vis_ranges[latest_img_idx]
            nh_high, nw_high = grid_dims[latest_img_idx]
            all_crop_info.append({
                "offset": offset_hi,
                "n_tokens": n_hi,
                "grid_h": nh_high,
                "grid_w": nw_high,
                "crop_bbox": crop_bbox,
            })

            if traj.round_actions[ri] == "click":
                if traj.click_token_index is None:
                    return None
                crop_vis_list = [
                    vis_embeds[ci["offset"]:ci["offset"] + ci["n_tokens"]]
                    for ci in all_crop_info
                ]
                combined_crop_vis = torch.cat(crop_vis_list, dim=0)
                _, _, click_logits = self.model.dual_head.click(combined_crop_vis, anchor)
                click_logits_1d = click_logits.squeeze(0) / temperature
                if traj.click_token_index >= click_logits_1d.shape[0]:
                    return None
                total_log_prob = total_log_prob + F.log_softmax(click_logits_1d, dim=-1)[
                    traj.click_token_index
                ]
                break

            _, _, logits = self.model.dual_head.look(vis_embeds, anchor, mask=full_mask)
            logits_1d = logits.squeeze(0) / temperature
            sampled_idx = traj.look_token_indices[ri]
            if sampled_idx is None or sampled_idx >= logits_1d.shape[0]:
                return None
            total_log_prob = total_log_prob + F.log_softmax(logits_1d, dim=-1)[sampled_idx]

        kl_penalty = torch.tensor(0.0, device=device)
        if self.ga.kl_coeff > 0 and self.ref_dual_head is not None:
            for p, p_ref in zip(
                self.model.dual_head.parameters(),
                self.ref_dual_head.parameters(),
            ):
                ref_param = p_ref.detach().to(device=device, dtype=p.dtype)
                kl_penalty = kl_penalty + ((p - ref_param) ** 2).sum()
            kl_penalty = self.ga.kl_coeff * kl_penalty

        # GRPO loss: -advantage * total_log_prob + KL penalty
        advantage = traj.reward
        loss = -advantage * total_log_prob + kl_penalty

        return loss

    # -- train step -----------------------------------------------------------

    def _train_step(self, sample):
        """Generate G trajectories, compute rewards, update."""
        device = _get_model_device(self.model)
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return 0.0

        # Generate G trajectories (with no_grad for sampling)
        trajectories = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(self.ga.group_size):
                traj = self._generate_trajectory(sample, img)
                if traj is not None:
                    trajectories.append(traj)

        if len(trajectories) < 2:
            return 0.0

        # Compute rewards and advantages
        self._compute_rewards(sample, trajectories)

        # Log metrics
        hits = sum(1 for t in trajectories if t.hit)
        avg_rounds = sum(t.n_rounds for t in trajectories) / len(trajectories)
        raw_rewards = [t.raw_reward for t in trajectories]
        avg_proximity = sum(t.proximity_score for t in trajectories) / len(trajectories)
        mean_reward = sum(raw_rewards) / len(raw_rewards)
        reward_std = (
            sum((r - mean_reward) ** 2 for r in raw_rewards) / len(raw_rewards)
        ) ** 0.5
        click_rounds = [
            next((ri + 1 for ri, action in enumerate(t.round_actions) if action == "click"), None)
            for t in trajectories
        ]
        click_rounds = [r for r in click_rounds if r is not None]
        no_click_rate = sum(1 for t in trajectories if t.click_token_index is None) / len(trajectories)
        avg_format_ok = sum(t.format_ok_rate for t in trajectories) / len(trajectories)
        avg_parse_fail = sum(t.parse_fail_rate for t in trajectories) / len(trajectories)
        avg_reason_tokens = sum(t.avg_reason_tokens for t in trajectories) / len(trajectories)
        total_actions = sum(len(t.round_actions) for t in trajectories)
        look_rate = (
            sum(action == "look" for t in trajectories for action in t.round_actions) / max(total_actions, 1)
        )
        click_rate = (
            sum(action == "click" for t in trajectories for action in t.round_actions) / max(total_actions, 1)
        )
        self.metrics["grpo_hit_rate"].append(hits / len(trajectories))
        self.metrics["grpo_avg_rounds"].append(avg_rounds)
        self.metrics["grpo_avg_reward"].append(
            sum(t.raw_reward for t in trajectories) / len(trajectories))
        self.metrics["grpo_avg_advantage"].append(
            sum(t.reward for t in trajectories) / len(trajectories))
        self.metrics["grpo_avg_proximity"].append(avg_proximity)
        self.metrics["grpo_reward_std"].append(reward_std)
        self.metrics["grpo_no_click_rate"].append(no_click_rate)
        self.metrics["grpo_format_ok_rate"].append(avg_format_ok)
        self.metrics["grpo_parse_fail_rate"].append(avg_parse_fail)
        self.metrics["grpo_avg_reason_tokens"].append(avg_reason_tokens)
        self.metrics["grpo_look_rate"].append(look_rate)
        self.metrics["grpo_click_rate"].append(click_rate)
        if click_rounds:
            self.metrics["grpo_avg_click_round"].append(
                sum(click_rounds) / len(click_rounds)
            )

        # Compute GRPO loss (differentiable replay)
        self.model.train()
        total_loss_value = 0.0

        for traj in trajectories:
            if abs(traj.reward) < 1e-8:
                continue
            traj_loss = self._replay_trajectory_loss(sample, traj)
            if traj_loss is not None:
                loss_for_backward = traj_loss / max(len(trajectories), 1)
                if loss_for_backward.requires_grad:
                    (loss_for_backward / max(self.args.gradient_accumulation_steps, 1)).backward()
                total_loss_value += traj_loss.item()

        return total_loss_value / max(len(trajectories), 1)

    # -- main train loop ------------------------------------------------------

    def train(self):
        bs = self.args.per_device_train_batch_size
        ga = max(self.args.gradient_accumulation_steps, 1)
        epochs = int(self.args.num_train_epochs)

        if self.rank == 0:
            print(f"=== GRPO Saccade Training ===")
            print(f"  samples={len(self.train_data)}  epochs={epochs}  bs={bs}  ga={ga}")
            print(f"  group_size={self.ga.group_size}  temperature={self.ga.temperature}")
            print(
                f"  reward_hit={self.ga.reward_hit}  proximity_weight={self.ga.reward_proximity_weight}  "
                f"format_reward={self.ga.reward_format}  malformed_penalty={self.ga.reward_malformed_penalty}  "
                f"round_penalty={self.ga.reward_round_penalty}"
            )
            print(f"  kl_coeff={self.ga.kl_coeff}  clip_eps={self.ga.clip_eps}")
            print(f"  head_lr={self.ga.action_head_lr}  backbone_lr={self.ga.lora_lr}")
            print(
                f"  max_saccade_rounds={self.ga.max_saccade_rounds}  "
                f"reasoning_max_new_tokens={self.ga.reasoning_max_new_tokens}  "
                f"use_dual_tokens={self.ga.use_dual_tokens}"
            )

        self.optimizer.zero_grad()
        micro = 0

        for epoch in range(epochs):
            epoch_rng = random.Random(42 + epoch)
            epoch_rng.shuffle(self.train_data)
            shard = self.train_data[self.rank::self.world_size] if self.world_size > 1 else self.train_data

            pbar = tqdm(range(0, len(shard), bs), desc=f"Epoch {epoch+1}/{epochs}",
                        disable=(self.rank != 0))
            for i in pbar:
                batch = shard[i:i+bs]
                batch_loss = 0.0
                for sample in batch:
                    try:
                        loss_val = self._train_step(sample)
                        batch_loss += loss_val
                    except Exception:
                        traceback.print_exc()

                if self.rank == 0:
                    def ar(k, n=20):
                        vals = self.metrics[k][-n:]
                        return sum(vals) / max(len(vals), 1)

                    postfix = {
                        "loss": f"{batch_loss / max(len(batch), 1):.3f}",
                    }
                    if self.metrics["grpo_hit_rate"]:
                        postfix["hit"] = f"{ar('grpo_hit_rate'):.1%}"
                        postfix["rounds"] = f"{ar('grpo_avg_rounds'):.2f}"
                        postfix["click_r"] = f"{ar('grpo_avg_click_round'):.2f}" if self.metrics["grpo_avg_click_round"] else "-"
                        postfix["no_click"] = f"{ar('grpo_no_click_rate'):.1%}"
                        postfix["fmt"] = f"{ar('grpo_format_ok_rate'):.1%}"
                        postfix["parse_fail"] = f"{ar('grpo_parse_fail_rate'):.1%}"
                        postfix["look"] = f"{ar('grpo_look_rate'):.1%}"
                        postfix["click"] = f"{ar('grpo_click_rate'):.1%}"
                        postfix["reason_toks"] = f"{ar('grpo_avg_reason_tokens'):.1f}"
                        postfix["prox"] = f"{ar('grpo_avg_proximity'):.3f}"
                        postfix["reward"] = f"{ar('grpo_avg_reward'):.3f}"
                        postfix["r_std"] = f"{ar('grpo_reward_std'):.3f}"
                        postfix["adv"] = f"{ar('grpo_avg_advantage'):.3f}"
                    if micro % ga == 0 and self.global_step > 0:
                        lrs = self.scheduler.get_last_lr()
                        postfix["step"] = str(self.global_step)
                        postfix["lr"] = f"{lrs[0]:.2e}"
                    pbar.set_postfix(postfix, refresh=False)

                micro += 1
                if micro % ga == 0:
                    if self.world_size > 1:
                        for p in self.model.parameters():
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    if any(p.grad is not None for p in self.model.parameters()):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Logging
                if self.rank == 0 and self.global_step % self.args.logging_steps == 0:
                    if micro % ga == 0 and self.metrics["grpo_hit_rate"]:
                        lrs = self.scheduler.get_last_lr()
                        click_round_str = (
                            f"{ar('grpo_avg_click_round'):.1f}"
                            if self.metrics["grpo_avg_click_round"]
                            else "-"
                        )
                        print(f"  [Step {self.global_step}] "
                              f"hit={ar('grpo_hit_rate'):.1%} "
                              f"rounds={ar('grpo_avg_rounds'):.1f} "
                              f"click_r={click_round_str} "
                              f"no_click={ar('grpo_no_click_rate'):.1%} "
                              f"fmt={ar('grpo_format_ok_rate'):.1%} "
                              f"parse_fail={ar('grpo_parse_fail_rate'):.1%} "
                              f"look={ar('grpo_look_rate'):.1%} "
                              f"click={ar('grpo_click_rate'):.1%} "
                              f"reason_toks={ar('grpo_avg_reason_tokens'):.1f} "
                              f"prox={ar('grpo_avg_proximity'):.3f} "
                              f"reward={ar('grpo_avg_reward'):.3f} "
                              f"r_std={ar('grpo_reward_std'):.3f} "
                              f"adv={ar('grpo_avg_advantage'):.3f} "
                              f"lr={lrs[0]:.2e}")
                        for key in list(self.metrics.keys()):
                            if len(self.metrics[key]) > 500:
                                self.metrics[key] = self.metrics[key][-500:]

                if self.global_step % self.args.save_steps == 0 and self.global_step > 0:
                    self._save(f"checkpoint-{self.global_step}")

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
        print(f"  Saved: {p}")


# -- Data loading (reuse from train.py) ----------------------------------------

def load_dataset(data_path, image_folder, max_samples=None, max_per_ds=None):
    from gui_attention.train import load_dataset as _load_dataset
    return _load_dataset(data_path, image_folder, max_samples, max_per_ds)


# -- Main ---------------------------------------------------------------------

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    rank = dist.get_rank() if dist.is_initialized() else 0

    parser = transformers.HfArgumentParser((GRPOArgs, GRPOTrainArgs))
    ga, ta = parser.parse_args_into_dataclasses()

    if rank == 0:
        print(f"Building model: {ga.model_name_or_path}")
    model, tokenizer, processor = build_model(
        ga.model_name_or_path,
        lora_r=ga.lora_r,
        lora_alpha=ga.lora_alpha,
        lora_target_modules=ga.lora_target_modules,
        torch_dtype=torch.bfloat16 if ta.bf16 else None,
        gradient_checkpointing=ta.gradient_checkpointing,
        use_lora=ga.use_lora,
        click_head_from=ga.click_head_from,
    )

    # Load SFT checkpoint if specified
    if ga.resume_ckpt:
        ckpt = ga.resume_ckpt
        if rank == 0:
            print(f"Loading SFT checkpoint: {ckpt}")
        if ga.use_lora:
            from peft import set_peft_model_state_dict
            adapter_file = os.path.join(ckpt, "adapter_model.safetensors")
            if os.path.exists(adapter_file):
                from safetensors.torch import load_file
                adapter_state = load_file(adapter_file)
            else:
                adapter_state = torch.load(os.path.join(ckpt, "adapter_model.bin"),
                                           map_location="cpu", weights_only=True)
            set_peft_model_state_dict(model.backbone, adapter_state)

        dual_path = os.path.join(ckpt, "dual_head.pt")
        old_path = os.path.join(ckpt, "action_head.pt")
        if os.path.exists(dual_path):
            model.dual_head.load_state_dict(
                torch.load(dual_path, map_location="cpu", weights_only=True))
        elif os.path.exists(old_path):
            old_state = torch.load(old_path, map_location="cpu", weights_only=True)
            look_state = {f"look_head.{k}": v for k, v in old_state.items()
                          if not k.startswith("bbox_head") and k != "beta"}
            model.dual_head.load_state_dict(look_state, strict=False)

    model.to(f"cuda:{local_rank}")

    # Reference dual head only (KL is applied only on dual_head params).
    ref_dual_head = deepcopy(model.dual_head).cpu().eval()
    for p in ref_dual_head.parameters():
        p.requires_grad = False

    train_data = load_dataset(ga.data_path, ga.image_folder, ga.max_samples, ga.max_samples_per_dataset)
    os.makedirs(ta.output_dir, exist_ok=True)

    trainer = GRPOTrainer(model, ref_dual_head, tokenizer, train_data, ta, ga)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
