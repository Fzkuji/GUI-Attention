"""
Multi-Precision Foveated Training (SFT with gui_aima attention).

Uses gui_aima's multi-layer multi-head attention aggregation with
visual-sink head weighting and IoU x Gaussian soft labels.

Each sample trains over max_rounds of progressive zoom (teacher forcing):
  Round 0: [system] [L0 full image] [instruction] [anchor]
  Round 1: [system] [L0 full image] [instruction] [L1 crop] [zoom desc] [anchor]
  Round 2: ... [L2 crop] [zoom desc] [anchor]
  Total loss = sum of per-round KL losses.

Usage:
    python -m gui_attention.train \
        --model_name_or_path /path/to/GUI-AIMA-3B \
        --data_path /path/to/guiact_bbox.json \
        --image_folder /path/to/images \
        --output_dir /path/to/checkpoints \
        ...
"""

import os
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image, ImageFile
from tqdm import tqdm

import transformers
from transformers import AutoProcessor, AutoTokenizer

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False

ImageFile.LOAD_TRUNCATED_IMAGES = True

from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.constants import ADDITIONAL_SPECIAL_TOKENS

from gui_attention.constants import (
    PRECISION_LEVELS, STOP_LEVELS, precision_for_level,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
)
from gui_attention.crop import crop_image
from gui_attention.attention import (
    find_image_visual_ranges,
    find_nth_pointer_pad,
    extract_attention,
    identify_attended_image,
    token_to_spatial,
)
from gui_attention.labels import compute_soft_labels
from gui_attention.builder import MultiRoundInputBuilder
from gui_attention.foveation import FoveationLoop


# -- Arguments ----------------------------------------------------------------

@dataclass
class ScriptArgs:
    model_name_or_path: str = field(default="/root/autodl-tmp/models/GUI-AIMA-3B")
    data_path: str = field(default=None)
    image_folder: str = field(default=None)
    max_samples: Optional[int] = field(default=None)
    min_pixels: int = field(default=3136)
    max_pixels: int = field(default=PRECISION_LEVELS[0])
    max_rounds: int = field(default=3)
    crop_ratio: float = field(default=0.3)
    initial_level: int = field(default=0, metadata={"help": "Precision level for round 0 (0-3)"})
    query_weighting: str = field(default="query_1", metadata={"help": "Visual-sink weighting strategy"})
    pointer_loss_weight: float = field(default=1.0)
    lm_loss_weight: float = field(default=0.0)
    sigma_scale: float = field(default=0.8, metadata={"help": "Gaussian sigma as fraction of bbox diagonal"})


@dataclass
class TrainArgs(transformers.TrainingArguments):
    training_mode: str = field(default="sft", metadata={"help": "'sft' teacher forcing"})
    optim: str = field(default="adamw_torch")
    gradient_checkpointing: bool = field(default=True)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)


# -- Reward -------------------------------------------------------------------

def position_reward(pred_x, pred_y, bbox_gt, img_w, img_h):
    gt_cx = (bbox_gt[0] + bbox_gt[2]) / 2
    gt_cy = (bbox_gt[1] + bbox_gt[3]) / 2
    dist_px = math.sqrt(((pred_x - gt_cx) * img_w) ** 2 +
                        ((pred_y - gt_cy) * img_h) ** 2)
    bbox_diag = max(math.sqrt(((bbox_gt[2] - bbox_gt[0]) * img_w) ** 2 +
                              ((bbox_gt[3] - bbox_gt[1]) * img_h) ** 2), 1.0)
    return 1.0 / (dist_px / bbox_diag + 1.0)


# -- Data ---------------------------------------------------------------------

def load_dataset(data_path, image_folder, max_samples=None):
    with open(data_path) as f:
        raw = json.load(f)
    if max_samples:
        random.shuffle(raw)
        raw = raw[:max_samples]
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
    print(f"Loaded {len(samples)} samples")
    return samples


# -- Helpers ------------------------------------------------------------------

def _get_model_device(model):
    if hasattr(model, 'hf_device_map'):
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, str):
            return torch.device(first_device)
        return torch.device(f"cuda:{first_device}")
    return model.device


def _get_all_visual_indices(input_ids, image_token_id, up_to_pos=None):
    """Get indices of ALL visual tokens, optionally only those before a given position."""
    ranges = find_image_visual_ranges(input_ids, image_token_id)
    indices = []
    range_offsets = []
    offset = 0
    for vs, ve in ranges:
        if up_to_pos is not None and vs >= up_to_pos:
            break
        indices.append(torch.arange(vs, ve, device=input_ids.device))
        range_offsets.append((offset, ve - vs))
        offset += ve - vs
    if not indices:
        return None, []
    return torch.cat(indices), range_offsets


def _get_query_indices(input_ids, image_token_id, pointer_pad_id, up_to_pos):
    """Get indices of text tokens (non-visual, non-pointer) before up_to_pos."""
    vis_set = set()
    ranges = find_image_visual_ranges(input_ids, image_token_id)
    for vs, ve in ranges:
        for i in range(vs, ve):
            vis_set.add(i)
    pp_set = set(pointer_pad_id) if isinstance(pointer_pad_id, list) else {pointer_pad_id}
    query_indices = []
    for i in range(up_to_pos):
        if i not in vis_set and input_ids[i].item() not in pp_set:
            query_indices.append(i)
    return query_indices


# -- Trainer ------------------------------------------------------------------

class MultiRoundTrainer:
    def __init__(self, model, processor, tokenizer, train_data,
                 args: TrainArgs, script_args: ScriptArgs):
        self.processor = processor
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.args = args
        self.sa = script_args
        self.use_deepspeed = HAS_DEEPSPEED and args.deepspeed is not None

        self.builder = MultiRoundInputBuilder(
            script_args.model_name_or_path, tokenizer, script_args.min_pixels,
        )
        self.fov_loop = FoveationLoop(
            max_rounds=script_args.max_rounds,
            crop_ratio=script_args.crop_ratio,
        )

        total_steps = (
            int(args.num_train_epochs) * len(train_data)
            // max(args.per_device_train_batch_size, 1)
            // max(args.gradient_accumulation_steps, 1)
        )
        if args.max_steps and args.max_steps > 0:
            total_steps = min(total_steps, args.max_steps)

        if self.use_deepspeed:
            self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
                model=model,
                model_parameters=[p for p in model.parameters() if p.requires_grad],
                config=args.deepspeed,
            )
            self.model = self.model_engine.module
            self.rank = dist.get_rank() if dist.is_initialized() else 0
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        else:
            self.model = model
            self.model_engine = None
            self.optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=args.learning_rate, weight_decay=args.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(total_steps, 1), eta_min=1e-7,
            )
            self.rank = 0
            self.world_size = 1

        self.metrics = defaultdict(list)
        self.global_step = 0

    # -- SFT forward: teacher forcing through precision levels ----------------

    def _sft_forward(self, sample):
        """SFT: teacher forcing with GT coords through precision levels.

        Builds rounds L0 -> L1 -> L2 with crops around GT.
        For each round, does a full forward pass, extracts multi-layer attention
        via gui_aima, generates IoU x Gaussian soft labels, and computes KL loss.

        Returns (loss, num_rounds, round_preds).
        """
        device = _get_model_device(self.model)
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception:
            return torch.tensor(0.0, device=device), 0, []

        bbox_gt = sample["bbox_gt"]
        gt_x = (bbox_gt[0] + bbox_gt[2]) / 2
        gt_y = (bbox_gt[1] + bbox_gt[3]) / 2

        # Build all rounds with teacher forcing
        self.builder.reset()
        level = self.sa.initial_level
        r0_inputs, cur_text, cur_images = self.builder.build_round0(
            sample, sample["instruction"], level=level,
        )

        # Collect per-round inputs
        round_inputs = [r0_inputs]
        num_sft_rounds = min(self.sa.max_rounds, len(PRECISION_LEVELS) - level)

        for ri in range(1, num_sft_rounds):
            next_level = level + ri
            if next_level >= len(PRECISION_LEVELS):
                break
            cropped, cbbox = crop_image(img, gt_x, gt_y, self.sa.crop_ratio)
            try:
                ri_inputs, cur_text, cur_images = self.builder.extend_with_crop(
                    cur_text, cur_images, cropped, cbbox, level=next_level,
                )
            except Exception:
                break
            round_inputs.append(ri_inputs)
            if next_level in STOP_LEVELS:
                break

        num_rounds = len(round_inputs)

        # Process each round independently
        self.model.train()
        total_loss = torch.tensor(0.0, device=device)
        n_valid = 0
        round_preds = []

        img_tok = self.model.config.image_token_id
        pp_id = self.model.config.pointer_pad_token_id

        for ri in range(num_rounds):
            inp = {k: v.to(device) for k, v in round_inputs[ri].items()}
            input_ids = inp["input_ids"]
            attention_mask = inp.get("attention_mask", torch.ones_like(input_ids))

            # Get position IDs
            position_ids, _ = self.model.get_rope_index(
                input_ids=input_ids, image_grid_thw=inp.get("image_grid_thw"),
                video_grid_thw=None, attention_mask=attention_mask,
            )

            # Forward pass
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask,
                pixel_values=inp.get("pixel_values"),
                image_grid_thw=inp.get("image_grid_thw"),
                output_hidden_states=True,
            )

            # Find pointer_pad for this round (it's always the last one in this input)
            ptr_pos = find_nth_pointer_pad(input_ids[0], pp_id, ri)
            if ptr_pos is None:
                round_preds.append(None)
                continue

            # Visual and query indices
            visual_indices, range_offsets = _get_all_visual_indices(
                input_ids[0], img_tok, up_to_pos=ptr_pos,
            )
            if visual_indices is None:
                round_preds.append(None)
                continue

            query_indices = _get_query_indices(
                input_ids[0], img_tok, pp_id, up_to_pos=ptr_pos,
            )

            # Extract multi-layer attention
            attn_weights, _ = extract_attention(
                self.model, outputs, input_ids, position_ids, attention_mask,
                visual_indices=visual_indices.tolist(),
                query_indices=query_indices,
                target_index=ptr_pos,
            )
            # attn_weights: (1, n_vis_total)

            # Determine which image is the highest-level (current round's latest image = image ri)
            vis_ranges = find_image_visual_ranges(input_ids[0], img_tok)
            vis_ranges_before_ptr = [(vs, ve) for vs, ve in vis_ranges if vs < ptr_pos]
            target_img_idx = min(ri, len(vis_ranges_before_ptr) - 1)

            # Get grid dims for the target image
            merge = self.model.visual.spatial_merge_size
            grid_dims = self.builder.get_image_grid_dims(inp["image_grid_thw"], merge)
            if target_img_idx >= len(grid_dims):
                round_preds.append(None)
                continue
            nh, nw = grid_dims[target_img_idx]

            # Generate soft labels for the target image's visual tokens
            # GT bbox in the local coordinate system of this image
            info = self.builder.image_infos[target_img_idx]
            bx1, by1, bx2, by2 = info.global_bbox
            bw = bx2 - bx1
            bh = by2 - by1
            if bw <= 0 or bh <= 0:
                round_preds.append(None)
                continue

            # Map GT bbox to local coords of this image
            local_gt_x1 = (bbox_gt[0] - bx1) / bw
            local_gt_y1 = (bbox_gt[1] - by1) / bh
            local_gt_x2 = (bbox_gt[2] - bx1) / bw
            local_gt_y2 = (bbox_gt[3] - by1) / bh
            # Clamp to [0, 1]
            local_gt_x1 = max(0.0, min(1.0, local_gt_x1))
            local_gt_y1 = max(0.0, min(1.0, local_gt_y1))
            local_gt_x2 = max(0.0, min(1.0, local_gt_x2))
            local_gt_y2 = max(0.0, min(1.0, local_gt_y2))

            local_gt_bbox = (local_gt_x1, local_gt_y1, local_gt_x2, local_gt_y2)
            soft_labels = compute_soft_labels(nh, nw, local_gt_bbox, self.sa.sigma_scale)
            soft_labels = soft_labels.to(device)

            # Build full label vector: zeros for other images, soft_labels for target image
            n_vis_total = attn_weights.shape[1]
            full_labels = torch.zeros(1, n_vis_total, device=device)
            if target_img_idx < len(range_offsets):
                offset = range_offsets[target_img_idx][0]
                n_tokens = range_offsets[target_img_idx][1]
                # soft_labels should be n_tokens long
                if soft_labels.numel() == n_tokens:
                    full_labels[0, offset:offset + n_tokens] = soft_labels
                else:
                    # Grid mismatch fallback: place at closest token
                    local_gt_cx = (local_gt_x1 + local_gt_x2) / 2
                    local_gt_cy = (local_gt_y1 + local_gt_y2) / 2
                    closest_col = min(max(int(local_gt_cx * nw), 0), nw - 1)
                    closest_row = min(max(int(local_gt_cy * nh), 0), nh - 1)
                    target_local = closest_row * nw + closest_col
                    if offset + target_local < n_vis_total:
                        full_labels[0, offset + target_local] = 1.0

            # Normalise full_labels
            label_sum = full_labels.sum()
            if label_sum > 0:
                full_labels = full_labels / label_sum

            # KL divergence loss
            epsilon = 1e-8
            pred_log = torch.log(attn_weights + epsilon)
            kl_loss = F.kl_div(pred_log, full_labels, reduction='batchmean')

            total_loss = total_loss + kl_loss
            n_valid += 1

            # Argmax prediction for metrics
            with torch.no_grad():
                attn_1d = attn_weights.squeeze(0)
                img_idx_pred, local_idx = identify_attended_image(
                    attn_1d, [(0, ro[1]) for ro in range_offsets],
                )
                if img_idx_pred < len(grid_dims):
                    nh_a, nw_a = grid_dims[img_idx_pred]
                    lx, ly = token_to_spatial(local_idx, nw_a, nh_a)
                    pred_info = self.builder.image_infos[img_idx_pred]
                    pbx1, pby1, pbx2, pby2 = pred_info.global_bbox
                    ox = pbx1 + lx * (pbx2 - pbx1)
                    oy = pby1 + ly * (pby2 - pby1)
                    round_preds.append((ox, oy))
                else:
                    round_preds.append(None)

        if n_valid > 0:
            total_loss = total_loss / n_valid
        return total_loss, n_valid, round_preds

    # -- train step -----------------------------------------------------------

    def _sft_train_step(self, sample):
        loss, nv, round_preds = self._sft_forward(sample)
        if nv > 0 and loss.requires_grad:
            if self.use_deepspeed:
                self.model_engine.backward(loss)
            else:
                (loss / max(self.args.gradient_accumulation_steps, 1)).backward()

        gcx = (sample["bbox_gt"][0] + sample["bbox_gt"][2]) / 2
        gcy = (sample["bbox_gt"][1] + sample["bbox_gt"][3]) / 2
        self.metrics["avg_rounds"].append(len(round_preds))
        valid_preds = [p for p in round_preds if p is not None]
        if valid_preds:
            final_px, final_py = valid_preds[-1]
            d = math.sqrt((final_px - gcx)**2 + (final_py - gcy)**2)
            self.metrics["avg_dist"].append(d)
            hit = (sample["bbox_gt"][0] <= final_px <= sample["bbox_gt"][2]
                   and sample["bbox_gt"][1] <= final_py <= sample["bbox_gt"][3])
            self.metrics["hit_rate"].append(1 if hit else 0)
        return loss.item() if nv > 0 else 0.0, nv

    def train_step(self, batch):
        acc_loss = 0.0
        acc_n = 0
        for sample in batch:
            try:
                loss_val, nv = self._sft_train_step(sample)
                if nv > 0:
                    acc_loss += loss_val
                    acc_n += 1
            except Exception as e:
                import traceback; traceback.print_exc()
        return acc_loss, acc_n

    # -- main train loop ------------------------------------------------------

    def train(self):
        bs = self.args.per_device_train_batch_size
        ga = max(self.args.gradient_accumulation_steps, 1)
        epochs = int(self.args.num_train_epochs)
        max_steps = getattr(self.args, 'max_steps', -1) or -1

        if self.rank == 0:
            print(f"=== Multi-Precision Foveated SFT (gui_aima attention) ===")
            print(f"  samples={len(self.train_data)}  epochs={epochs}  bs={bs}  ga={ga}")
            print(f"  max_rounds={self.sa.max_rounds}  initial_level={self.sa.initial_level}")
            print(f"  levels={PRECISION_LEVELS}  stop_at={STOP_LEVELS}")
            print(f"  crop_ratio={self.sa.crop_ratio}  lr={self.args.learning_rate}")
            print(f"  query_weighting={self.sa.query_weighting}")
            print(f"  pointer_loss_weight={self.sa.pointer_loss_weight}")
            print(f"  lm_loss_weight={self.sa.lm_loss_weight}")
            print(f"  sigma_scale={self.sa.sigma_scale}")
            print(f"  deepspeed={self.use_deepspeed}  max_steps={max_steps}")

        if not self.use_deepspeed:
            self.optimizer.zero_grad()
        micro = 0

        for epoch in range(epochs):
            random.shuffle(self.train_data)
            if self.world_size > 1:
                shard = self.train_data[self.rank::self.world_size]
            else:
                shard = self.train_data
            pbar = tqdm(range(0, len(shard), bs), desc=f"Epoch {epoch+1}/{epochs}",
                        disable=(self.rank != 0))
            for i in pbar:
                loss_val, nv = self.train_step(shard[i:i+bs])
                micro += 1
                if self.use_deepspeed:
                    self.model_engine.step()
                    self.global_step += 1
                elif micro % ga == 0:
                    if any(p.grad is not None for p in self.model.parameters()):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Logging
                if self.rank == 0 and self.global_step % self.args.logging_steps == 0:
                    should_log = self.use_deepspeed or (micro % ga == 0)
                    if should_log:
                        ar = lambda k: sum(self.metrics[k][-20:]) / max(len(self.metrics[k][-20:]), 1)
                        lr_val = (self.model_engine.get_lr()[0] if self.use_deepspeed
                                  else self.scheduler.get_last_lr()[0])
                        print(f"  [Step {self.global_step}] loss={loss_val:.4f} "
                              f"hit={ar('hit_rate'):.1%} dist={ar('avg_dist'):.4f} "
                              f"rounds={ar('avg_rounds'):.1f} lr={lr_val:.2e}")
                        for key in list(self.metrics.keys()):
                            if len(self.metrics[key]) > 200:
                                self.metrics[key] = self.metrics[key][-200:]

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
        if self.rank != 0:
            return
        p = os.path.join(self.args.output_dir, name)
        os.makedirs(p, exist_ok=True)
        self.model.save_pretrained(p)
        self.tokenizer.save_pretrained(p)
        self.processor.save_pretrained(p)
        with open(os.path.join(p, "metrics.json"), "w") as f:
            json.dump({k: v[-100:] for k, v in self.metrics.items()}, f)
        print(f"  Saved: {p}")


# -- Main --------------------------------------------------------------------

def main():
    parser = transformers.HfArgumentParser((ScriptArgs, TrainArgs))
    sa, ta = parser.parse_args_into_dataclasses()

    print(f"Loading model: {sa.model_name_or_path}")
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        sa.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if ta.bf16 else None,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False
    if ta.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad = True

    # Configure attention args
    model.set_attention_args(sa.query_weighting)
    model.reset_loss_weights(sa.pointer_loss_weight, sa.lm_loss_weight)

    tokenizer = AutoTokenizer.from_pretrained(sa.model_name_or_path)
    processor = AutoProcessor.from_pretrained(
        sa.model_name_or_path, min_pixels=sa.min_pixels, max_pixels=sa.max_pixels,
    )
    processor.tokenizer = tokenizer

    # Pointer token IDs should already be in the model/tokenizer for GUI-AIMA
    # but set them explicitly if missing
    if not hasattr(model.config, 'pointer_pad_token_id') or model.config.pointer_pad_token_id is None:
        num_new = tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
        if num_new > 0:
            model.resize_token_embeddings(len(tokenizer))
            ie = model.get_input_embeddings().weight.data
            oe = model.get_output_embeddings().weight.data
            ie[-num_new:] = ie[:-num_new].mean(0, keepdim=True)
            oe[-num_new:] = oe[:-num_new].mean(0, keepdim=True)
        model.config.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
        model.config.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
        model.config.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]

    if not (HAS_DEEPSPEED and ta.deepspeed):
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_data = load_dataset(sa.data_path, sa.image_folder, sa.max_samples)
    os.makedirs(ta.output_dir, exist_ok=True)

    trainer = MultiRoundTrainer(model, processor, tokenizer, train_data, ta, sa)
    trainer.train()


if __name__ == "__main__":
    main()
