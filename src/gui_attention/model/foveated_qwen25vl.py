"""
Foveated Qwen2.5-VL for GUI Grounding.

Two-phase approach:
1. Pointer-token attention training (from GUI-AIMA): trains model attention to
   focus on target UI elements via special pointer tokens + attention supervision.
2. Iterative foveated inference: progressive zoom-in using trained attention
   (periphery → parafovea → fovea) to achieve efficient grounding.

Key components:
- calculate_attention_from_qk: QK-recompute with RoPE for clean attention extraction
- GroundingHead: Query-weighted multi-head attention aggregation
- Pointer tokens: <pointer_start>, <pointer_pad>, <pointer_end> for attention anchor
- Attention supervision: KL-div loss forcing pointer_pad attention onto bbox patches
"""

import math
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLCausalLMOutputWithPast,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from gui_attention.foveation.sampler import FoveatedSampler


# Token IDs
IMAGE_PAD_ID = 151655  # <|image_pad|>

# Special pointer tokens
POINTER_START_TOKEN = "<|pointer_start|>"
POINTER_PAD_TOKEN = "<|pointer_pad|>"
POINTER_END_TOKEN = "<|pointer_end|>"


def calculate_attention_from_qk(
    model, all_hidden_states, all_position_ids=None,
    all_attention_mask=None, query_indices=None,
):
    """Recompute QK attention from hidden states with RoPE.
    Ported from GUI-AIMA for clean, semantically meaningful attention extraction."""
    qwen_decoder = model.model
    num_layers = len(qwen_decoder.layers)
    all_timesteps_attention = []

    for t, hs_per_layer in enumerate(all_hidden_states):
        bsz, seq_len, _ = hs_per_layer[0].shape
        q_idx = query_indices if query_indices is not None else [seq_len - 1]

        position_ids = all_position_ids if torch.is_tensor(all_position_ids) else all_position_ids[t]
        cos, sin = qwen_decoder.rotary_emb(hs_per_layer[0], position_ids)

        # Causal mask
        if all_attention_mask is not None:
            attn_mask_2d = all_attention_mask if torch.is_tensor(all_attention_mask) else all_attention_mask[t]
            orig_impl = qwen_decoder.config._attn_implementation
            qwen_decoder.config._attn_implementation = "eager"
            causal_mask = qwen_decoder._update_causal_mask(
                attn_mask_2d, hs_per_layer[0],
                cache_position=torch.arange(seq_len, device=hs_per_layer[0].device),
                past_key_values=None, output_attentions=True,
            )
            qwen_decoder.config._attn_implementation = orig_impl
        else:
            causal_mask = None

        timestep_attns = []
        for layer_idx in range(num_layers):
            layer = qwen_decoder.layers[layer_idx]
            sa = layer.self_attn
            li = layer.input_layernorm(hs_per_layer[layer_idx])

            k = sa.k_proj(li).view(bsz, seq_len, -1, sa.head_dim).transpose(1, 2)
            q = sa.q_proj(li[:, q_idx, :]).view(bsz, len(q_idx), -1, sa.head_dim).transpose(1, 2)
            k = repeat_kv(k, sa.num_key_value_groups)

            k, _ = apply_multimodal_rotary_pos_emb(k, k.clone(), cos, sin, sa.rope_scaling["mrope_section"])
            cos_q, sin_q = cos[:, :, q_idx, :], sin[:, :, q_idx, :]
            q, _ = apply_multimodal_rotary_pos_emb(q, q.clone(), cos_q, sin_q, sa.rope_scaling["mrope_section"])

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(sa.head_dim)
            if causal_mask is not None:
                scores = scores + causal_mask[:, :, q_idx, :].to(scores.dtype)
            timestep_attns.append(F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype))

        all_timesteps_attention.append(timestep_attns)
    return all_timesteps_attention


def get_prediction_from_attention(attn_scores, n_width, n_height, activation_threshold=0.3):
    """Extract prediction point from attention heatmap using connected region analysis."""
    max_score = attn_scores[0].max().item()
    if max_score < 1e-10:
        return 0.5, 0.5

    threshold = max_score * activation_threshold
    mask = attn_scores[0] > threshold
    valid_indices = torch.nonzero(mask).squeeze(-1)
    if len(valid_indices) == 0:
        return 0.5, 0.5

    topk_values = attn_scores[0][valid_indices]
    topk_coords = [(idx.item() // n_width, idx.item() % n_width, idx.item()) for idx in valid_indices]

    # BFS connected regions
    regions = []
    visited = set()
    for i, (y, x, idx) in enumerate(topk_coords):
        if idx in visited:
            continue
        region = [(y, x, idx, topk_values[i].item())]
        visited.add(idx)
        queue = [(y, x, idx, topk_values[i].item())]
        while queue:
            cy, cx, _, _ = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                n_idx = ny * n_width + nx
                for j, (ty, tx, t_idx) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, topk_values[j].item()))
                        queue.append((ny, nx, t_idx, topk_values[j].item()))
        regions.append(region)

    best_region = max(regions, key=lambda r: max(item[3] for item in r))
    total_w = sum(s for _, _, _, s in best_region)
    wx = sum(((x + 0.5) / n_width) * s for _, x, _, s in best_region) / total_w
    wy = sum(((y + 0.5) / n_height) * s for y, _, _, s in best_region) / total_w
    return wx, wy


class GroundingHead(nn.Module):
    """Multi-head attention aggregation for grounding.
    Computes weighted attention from pointer_pad token to visual tokens,
    using query-token importance for head weighting."""

    def __init__(self, query_topk=5):
        super().__init__()
        self.query_topk = query_topk

    def forward(self, query_indices, visual_indices, target_indices,
                self_attentions, topk_query_indices, batch_idx=0,
                labels=None):
        """
        Args:
            self_attentions: list of (B, H, Q, seq_len) per layer
            labels: (1, n_visual) binary mask for supervision
        Returns:
            attn_weights: (1, n_visual) normalized attention
            loss: KL-div loss if labels provided
        """
        epsilon = 1e-8
        all_head_attns = []
        query_head_attns = []

        for layer_idx in range(len(self_attentions)):
            layer_attn = self_attentions[layer_idx][batch_idx]  # (H, Q, seq_len)
            # Target (pointer_pad) → visual tokens
            target_attn = layer_attn[:, -1, visual_indices]  # (H, n_visual)
            all_head_attns.append(target_attn)
            # Query → visual for head weighting
            if topk_query_indices is not None:
                q_attn = layer_attn[:, topk_query_indices, :][:, :, visual_indices]
                query_head_attns.append(q_attn)

        all_head_attns_cat = torch.cat(all_head_attns, dim=0)  # (L*H, n_visual)

        if topk_query_indices is not None and query_head_attns:
            query_cat = torch.cat(query_head_attns, dim=0)
            head_weights = query_cat.sum(dim=(-1, -2))
            head_weights = head_weights.softmax(dim=-1)
            attn_merged = torch.mul(head_weights[:, None], all_head_attns_cat)
            attn_merged = attn_merged.sum(dim=0, keepdim=True)
        else:
            attn_merged = all_head_attns_cat.mean(dim=0, keepdim=True)

        attn_merged = attn_merged / (attn_merged.sum(dim=-1, keepdim=True) + epsilon)

        loss = None
        if labels is not None:
            target_dist = labels.float()
            target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + epsilon)
            pred_log = torch.log(attn_merged + epsilon)
            loss = F.kl_div(pred_log, target_dist, reduction='batchmean')

        return attn_merged, loss


class OutputWithPointerLoss(Qwen2_5_VLCausalLMOutputWithPast):
    def __init__(self, lm_loss=None, pointer_loss=None, pointer_scores=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lm_loss = lm_loss
        self.pointer_loss = pointer_loss
        self.pointer_scores = pointer_scores


class Qwen25VLWithPointer(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL extended with pointer-token attention training.

    Adds:
    - Special pointer tokens for attention anchoring
    - Attention supervision loss (KL-div on pointer → visual attention)
    - Query-weighted multi-head attention aggregation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grounding_head = GroundingHead(query_topk=5)
        self.pointer_loss_weight = 1.0
        self.lm_loss_weight = 1.0
        self.post_init()

    def forward(
        self,
        input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None,
        use_cache=None, output_attentions=None, output_hidden_states=None,
        return_dict=None, pixel_values=None, pixel_values_videos=None,
        image_grid_thw=None, video_grid_thw=None, rope_deltas=None,
        cache_position=None, second_per_grid_ts=None,
        # Grounding supervision
        multi_patch_labels=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Standard embedding
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask.to(inputs_embeds.device),
                                                             image_embeds.to(inputs_embeds.device, inputs_embeds.dtype))
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # Position IDs
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None or \
               (past_key_values is None or past_key_values.get_seq_length() == 0):
                position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
                self.rope_deltas = rope_deltas

        # Forward through transformer
        outputs = self.model(
            input_ids=None, position_ids=position_ids, attention_mask=attention_mask,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            use_cache=use_cache, output_attentions=False, output_hidden_states=True,
            return_dict=return_dict, cache_position=cache_position,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # LM loss
        lm_loss = None
        if labels is not None and self.lm_loss_weight > 0:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size),
                                             shift_labels.view(-1).to(shift_logits.device))

        # Pointer attention loss
        pointer_loss = None
        pointer_scores = []
        if multi_patch_labels is not None:
            batch_size = input_ids.shape[0]
            pointer_losses = []

            for i in range(batch_size):
                token_ids = input_ids[i]
                visual_mask = (token_ids == self.config.image_token_id)
                visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1)

                pointer_pad_id = self.config.pointer_pad_token_id
                if isinstance(pointer_pad_id, list):
                    target_mask = torch.isin(token_ids, torch.tensor(pointer_pad_id, device=token_ids.device))
                else:
                    target_mask = (token_ids == pointer_pad_id)
                target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)

                if visual_indices.numel() == 0 or target_indices.numel() == 0:
                    continue

                # Query indices: text tokens between last visual and pointer_start
                query_start = visual_indices[-1].item() + 1
                pointer_start_id = self.config.pointer_start_token_id
                ps_mask = (token_ids == pointer_start_id)
                ps_positions = torch.nonzero(ps_mask, as_tuple=False).squeeze(-1)
                query_end = ps_positions[0].item() if ps_positions.numel() > 0 else len(token_ids)
                query_indices = torch.arange(query_start, query_end, device=token_ids.device)

                merged_indices = torch.cat([query_indices, target_indices], dim=0)

                # QK-recompute attention
                calc_attn = calculate_attention_from_qk(
                    model=self, all_hidden_states=[outputs.hidden_states],
                    all_position_ids=position_ids, query_indices=merged_indices,
                    all_attention_mask=attention_mask,
                )

                # Query importance (cosine similarity)
                all_layer_hs = torch.stack(outputs.hidden_states[1:], dim=0)
                sample_hs = all_layer_hs[:, i, :, :]
                q_hs = F.normalize(sample_hs[:, query_indices, :], dim=-1)
                v_hs = F.normalize(sample_hs[:, visual_indices, :], dim=-1)
                sim = torch.einsum('lqd,lvd->lqv', q_hs, v_hs)
                agg = sim.sum(dim=-1).sum(dim=0)  # (n_query,)
                k = min(self.grounding_head.query_topk, len(query_indices))
                _, topk_idx = torch.topk(agg, k, largest=True)

                # Grounding head
                attn_scores, loss_v = self.grounding_head(
                    query_indices, visual_indices, target_indices,
                    calc_attn[0], topk_idx, batch_idx=i,
                    labels=multi_patch_labels[i],
                )
                pointer_scores.append(attn_scores.detach().cpu())
                pointer_losses.append(loss_v)

            if pointer_losses:
                pointer_loss = torch.stack(pointer_losses).mean()

        # Combined loss
        total_loss = None
        if lm_loss is not None and pointer_loss is not None:
            total_loss = self.lm_loss_weight * lm_loss + self.pointer_loss_weight * pointer_loss
        elif lm_loss is not None:
            total_loss = lm_loss
        elif pointer_loss is not None:
            total_loss = pointer_loss

        if return_dict:
            return OutputWithPointerLoss(
                lm_loss=lm_loss, pointer_loss=pointer_loss, pointer_scores=pointer_scores,
                loss=total_loss, logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states, attentions=None,
                rope_deltas=self.rope_deltas,
            )
        return (total_loss, logits) + outputs[1:]


class FoveatedQwen25VL(nn.Module):
    """Wrapper for iterative foveated inference using a trained Qwen25VLWithPointer model.

    Inference flow:
        Round 1: Low-res full image → pointer attention → rough fixation
        Round 2: Medium-res crop at fixation → refine
        Round 3: High-res crop → final prediction
    """

    def __init__(self, model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct", fovea_config=None, query_topk=5):
        super().__init__()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
        )
        self.sampler = FoveatedSampler(**(fovea_config or {}))
        self.query_topk = query_topk
        for p in self.model.parameters():
            p.requires_grad = False

    def _single_round_inference(self, image, instruction, crop_bbox=(0., 0., 1., 1.), max_new_tokens=50):
        """Run one round: generate + QK-recompute attention → coordinates."""
        from qwen_vl_utils import process_vision_info

        conversation = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": instruction},
        ]}]
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(conversation)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(self.model.device)

        input_ids = inputs["input_ids"][0]
        with torch.no_grad():
            position_ids, _ = self.model.get_rope_index(
                input_ids=inputs["input_ids"], image_grid_thw=inputs["image_grid_thw"],
                video_grid_thw=None, attention_mask=inputs["attention_mask"],
            )
            results = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                return_dict_in_generate=True, output_hidden_states=True, temperature=0.1,
            )

        generated_ids = results.sequences[0][len(input_ids):]
        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        visual_mask = input_ids == IMAGE_PAD_ID
        visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1)
        n_visual = visual_indices.shape[0]
        if n_visual == 0:
            return 0.5, 0.5, {"text": generated_text}

        query_start = visual_indices[-1].item() + 1
        query_end = len(input_ids)
        query_indices = torch.arange(query_start, query_end, device=input_ids.device)
        target_idx = torch.tensor([len(input_ids) - 1], device=input_ids.device)
        merged = torch.cat([query_indices, target_idx], dim=0)

        calc_attn = calculate_attention_from_qk(
            model=self.model, all_hidden_states=[results.hidden_states[0]],
            all_position_ids=position_ids, query_indices=merged,
            all_attention_mask=inputs["attention_mask"],
        )

        # Query weighting
        all_hs = torch.stack(results.hidden_states[0][1:], dim=0)[:, 0, :, :]
        q_hs = F.normalize(all_hs[:, query_indices, :], dim=-1)
        v_hs = F.normalize(all_hs[:, visual_indices, :], dim=-1)
        agg = torch.einsum('lqd,lvd->lqv', q_hs, v_hs).sum(dim=-1).sum(dim=0)
        k = min(self.query_topk, len(query_indices))
        _, topk_idx = torch.topk(agg, k, largest=True)

        # Aggregate attention
        num_layers = len(calc_attn[0])
        all_head_attns, query_head_attns = [], []
        for li in range(num_layers):
            la = calc_attn[0][li][0]  # (H, Q, seq)
            all_head_attns.append(la[:, -1, visual_indices])
            query_head_attns.append(la[:, topk_idx, :][:, :, visual_indices])

        ahc = torch.cat(all_head_attns, dim=0)
        qhc = torch.cat(query_head_attns, dim=0)
        hw = qhc.sum(dim=(-1, -2)).softmax(dim=-1)
        attn = (hw[:, None] * ahc).sum(dim=0, keepdim=True)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        merge = self.model.config.vision_config.spatial_merge_size
        _, nh, nw = (inputs["image_grid_thw"][0] // merge).tolist()
        px, py = get_prediction_from_attention(attn, int(nw), int(nh))

        x1, y1, x2, y2 = crop_bbox
        ox, oy = x1 + px * (x2 - x1), y1 + py * (y2 - y1)
        return ox, oy, {"text": generated_text, "attn_max": attn.max().item(), "attn_min": attn.min().item()}

    @torch.no_grad()
    def predict(self, image, instruction, fixation=None, num_rounds=3, max_new_tokens=50):
        """Iterative foveated inference."""
        from PIL import Image as PILImage
        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")
        W, H = image.size

        # Round 1: Periphery
        ph = int(self.sampler.periphery_resolution * H / W)
        periphery = image.resize((self.sampler.periphery_resolution, ph), PILImage.LANCZOS)
        fx, fy, d1 = self._single_round_inference(periphery, instruction, (0., 0., 1., 1.), max_new_tokens)
        print(f"  R1 (periphery): pred=({fx:.4f},{fy:.4f}) attn=[{d1.get('attn_min',0):.6f},{d1.get('attn_max',0):.6f}]")
        if fixation: fx, fy = fixation
        if num_rounds < 2: return (fx, fy)

        # Round 2: Parafovea
        para = self.sampler._extract_crop(image, fx, fy, self.sampler.parafovea_size, self.sampler.parafovea_resolution)
        fx, fy, d2 = self._single_round_inference(para["image"], instruction, para["bbox"], max_new_tokens)
        print(f"  R2 (parafovea): pred=({fx:.4f},{fy:.4f}) attn=[{d2.get('attn_min',0):.6f},{d2.get('attn_max',0):.6f}]")
        if num_rounds < 3: return (fx, fy)

        # Round 3: Fovea
        fovea = self.sampler._extract_crop(image, fx, fy, self.sampler.fovea_size, self.sampler.fovea_resolution)
        fx, fy, d3 = self._single_round_inference(fovea["image"], instruction, fovea["bbox"], max_new_tokens)
        print(f"  R3 (fovea): pred=({fx:.4f},{fy:.4f}) attn=[{d3.get('attn_min',0):.6f},{d3.get('attn_max',0):.6f}]")
        return (fx, fy)
