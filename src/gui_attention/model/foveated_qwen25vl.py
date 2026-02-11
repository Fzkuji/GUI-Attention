"""
Foveated Qwen2.5-VL for GUI Grounding.

Iterative foveated inference with GUI-AIMA-style attention extraction:
1. Progressive zoom-in (periphery → parafovea → fovea) mimicking human saccades
2. QK-recompute attention extraction with RoPE (from GUI-AIMA)
3. Query-weighted multi-head attention aggregation for grounding
4. Region-based prediction from attention heatmap
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from gui_attention.foveation.sampler import FoveatedSampler


# Qwen2.5-VL special token IDs
IMAGE_PAD_ID = 151655  # <|image_pad|>


def calculate_attention_from_qk(
    model,
    all_hidden_states,
    all_position_ids=None,
    all_attention_mask=None,
    query_indices=None,
):
    """Recompute QK attention from hidden states with RoPE.

    Ported from GUI-AIMA. This gives semantically meaningful attention
    unlike raw output_attentions from prefill.

    Args:
        model: Qwen2.5-VL model.
        all_hidden_states: List of per-timestep hidden states.
            Each element is a list of (B, seq_len, hidden) per layer.
        all_position_ids: Position IDs for RoPE (3, B, seq_len).
        all_attention_mask: Attention mask.
        query_indices: Which positions to compute attention FROM.

    Returns:
        List of per-timestep attention: each is a list of (B, H, Q, seq_len) per layer.
    """
    qwen_decoder = model.model
    num_layers = len(qwen_decoder.layers)
    all_timesteps_attention = []

    for t, hs_per_layer in enumerate(all_hidden_states):
        bsz, seq_len, _ = hs_per_layer[0].shape

        if query_indices is None:
            q_idx = [seq_len - 1]
        else:
            q_idx = query_indices

        if all_position_ids is not None:
            if torch.is_tensor(all_position_ids):
                position_ids = all_position_ids
            else:
                position_ids = all_position_ids[t]
        cos, sin = qwen_decoder.rotary_emb(hs_per_layer[0], position_ids)

        # Build causal mask
        if all_attention_mask is not None:
            if torch.is_tensor(all_attention_mask):
                attn_mask_2d = all_attention_mask
            else:
                attn_mask_2d = all_attention_mask[t]
            orig_impl = qwen_decoder.config._attn_implementation
            qwen_decoder.config._attn_implementation = "eager"
            causal_mask = qwen_decoder._update_causal_mask(
                attn_mask_2d,
                hs_per_layer[0],
                cache_position=torch.arange(seq_len, device=hs_per_layer[0].device),
                past_key_values=None,
                output_attentions=True,
            )
            qwen_decoder.config._attn_implementation = orig_impl
        else:
            causal_mask = None

        timestep_attns = []

        for layer_idx in range(num_layers):
            layer = qwen_decoder.layers[layer_idx]
            self_attn = layer.self_attn

            layer_input = hs_per_layer[layer_idx]
            layer_input = layer.input_layernorm(layer_input)
            layer_input_q = layer_input[:, q_idx, :]

            k_proj = self_attn.k_proj(layer_input)
            q_proj = self_attn.q_proj(layer_input_q)

            k = k_proj.view(bsz, seq_len, -1, self_attn.head_dim).transpose(1, 2)
            q = q_proj.view(bsz, len(q_idx), -1, self_attn.head_dim).transpose(1, 2)

            k = repeat_kv(k, self_attn.num_key_value_groups)

            # Apply RoPE
            k, _ = apply_multimodal_rotary_pos_emb(
                k, k.clone(), cos, sin, self_attn.rope_scaling["mrope_section"]
            )
            cos_q = cos[:, :, q_idx, :]
            sin_q = sin[:, :, q_idx, :]
            q, _ = apply_multimodal_rotary_pos_emb(
                q, q.clone(), cos_q, sin_q, self_attn.rope_scaling["mrope_section"]
            )

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self_attn.head_dim)

            if causal_mask is not None:
                attn_scores = attn_scores + causal_mask[:, :, q_idx, :].to(attn_scores.dtype)

            attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
            timestep_attns.append(attn_weights)

        all_timesteps_attention.append(timestep_attns)

    return all_timesteps_attention


def get_prediction_from_attention(
    attn_scores: torch.Tensor,
    n_width: int,
    n_height: int,
    activation_threshold: float = 0.3,
) -> tuple[float, float]:
    """Extract prediction point from attention heatmap over visual patches.

    Ported from GUI-AIMA's get_prediction_region_point. Finds connected
    regions of high-attention patches and returns the weighted center
    of the highest-scoring region.

    Args:
        attn_scores: (1, n_patches) attention weights.
        n_width: Number of patches in width.
        n_height: Number of patches in height.
        activation_threshold: Fraction of max attention to threshold.

    Returns:
        (x, y) normalized coordinates in [0, 1].
    """
    max_score = attn_scores[0].max().item()
    if max_score < 1e-10:
        return 0.5, 0.5

    threshold = max_score * activation_threshold
    mask = attn_scores[0] > threshold
    valid_indices = torch.nonzero(mask).squeeze(-1)

    if len(valid_indices) == 0:
        return 0.5, 0.5

    topk_values = attn_scores[0][valid_indices]

    # Convert to 2D coordinates
    topk_coords = []
    for idx in valid_indices.tolist():
        y = idx // n_width
        x = idx % n_width
        topk_coords.append((y, x, idx))

    # Find connected regions via BFS
    regions = []
    visited = set()
    for i, (y, x, idx) in enumerate(topk_coords):
        if idx in visited:
            continue
        region = [(y, x, idx, topk_values[i].item())]
        visited.add(idx)
        queue = [(y, x, idx, topk_values[i].item())]
        while queue:
            cy, cx, c_idx, c_val = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                n_idx = ny * n_width + nx
                for j, (ty, tx, t_idx) in enumerate(topk_coords):
                    if ty == ny and tx == nx and t_idx not in visited:
                        visited.add(t_idx)
                        region.append((ny, nx, t_idx, topk_values[j].item()))
                        queue.append((ny, nx, t_idx, topk_values[j].item()))
        regions.append(region)

    # Pick region with highest max score
    best_region = max(regions, key=lambda r: max(item[3] for item in r))

    # Weighted average of patch centers
    total_weight = sum(item[3] for item in best_region)
    weighted_x = sum(((x + 0.5) / n_width) * score for y, x, _, score in best_region) / total_weight
    weighted_y = sum(((y + 0.5) / n_height) * score for y, x, _, score in best_region) / total_weight

    return weighted_x, weighted_y


class FoveatedQwen25VL(nn.Module):
    """Qwen2.5-VL with iterative foveated attention for GUI grounding.

    Architecture (per round):
        Image crop → Qwen2.5-VL generate() → hidden states
                                                ↓
        calculate_attention_from_qk (QK recompute with RoPE)
                                                ↓
        Query-weighted attention aggregation over visual tokens
                                                ↓
        Region-based coordinate prediction from attention heatmap

    Iterative refinement:
        Round 1: periphery (full image, low-res) → find attention peak
        Round 2: parafovea crop around peak → refine
        Round 3: fovea crop around refined peak → final prediction
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        fovea_config: Optional[dict] = None,
        query_topk: int = 5,
    ):
        super().__init__()

        # Load base model
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        # Foveated sampler
        self.sampler = FoveatedSampler(**(fovea_config or {}))

        # Config for attention aggregation
        self.query_topk = query_topk

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

    def _single_round_inference(
        self,
        image,
        instruction: str,
        crop_bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
        max_new_tokens: int = 50,
    ) -> tuple[float, float, dict]:
        """Run one round of inference on a single image crop.

        Uses GUI-AIMA-style attention extraction:
        1. Generate tokens with output_hidden_states
        2. QK-recompute attention with RoPE
        3. Query-weighted attention aggregation
        4. Region-based coordinate prediction

        Args:
            image: PIL Image (the crop to process).
            instruction: Text instruction.
            crop_bbox: (x1, y1, x2, y2) of this crop in original image coords.
            max_new_tokens: Tokens to generate.

        Returns:
            (x, y) in original image normalized coordinates, plus debug info dict.
        """
        from qwen_vl_utils import process_vision_info

        # Build conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(conversation)
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        input_ids = inputs["input_ids"][0]

        # Get position_ids for RoPE
        with torch.no_grad():
            position_ids, _ = self.model.get_rope_index(
                input_ids=inputs["input_ids"],
                image_grid_thw=inputs["image_grid_thw"],
                video_grid_thw=None,
                attention_mask=inputs["attention_mask"],
            )

        # Generate with hidden states
        with torch.no_grad():
            results = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_hidden_states=True,
                temperature=0.1,
            )

        # Decode generated text
        generated_ids = results.sequences[0][len(input_ids):]
        generated_text = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        # Find visual token indices
        visual_mask = input_ids == IMAGE_PAD_ID
        visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1)
        n_visual = visual_indices.shape[0]

        if n_visual == 0:
            return 0.5, 0.5, {"text": generated_text, "n_visual": 0}

        # Find query indices: text tokens between last visual token and end of input
        # These are the instruction tokens that carry semantic meaning
        query_start = visual_indices[-1].item() + 1
        query_end = len(input_ids)
        query_indices = torch.arange(query_start, query_end, device=input_ids.device)

        # Use the last generated token as "target" (like pointer_pad in GUI-AIMA)
        # We use the last token position in generation
        # For QK recompute, we need to use prefill hidden states
        # The target is the last query token (analogous to pointer_pad)
        # Merge query_indices with a "target" index (last input token)
        target_idx = torch.tensor([len(input_ids) - 1], device=input_ids.device)
        merged_indices = torch.cat([query_indices, target_idx], dim=0)

        # QK-recompute attention from hidden states
        calculated_attention = calculate_attention_from_qk(
            model=self.model,
            all_hidden_states=results.hidden_states,
            all_position_ids=position_ids,
            query_indices=merged_indices,
            all_attention_mask=inputs["attention_mask"],
        )

        # Get hidden states for query weighting (cosine similarity)
        all_layer_hs = torch.stack(results.hidden_states[0][1:], dim=0)  # (n_layer, B, seq, d)
        sample_layer_hs = all_layer_hs[:, 0, :, :]  # (n_layer, seq, d)

        query_hs = sample_layer_hs[:, query_indices, :]  # (n_layer, n_query, d)
        visual_hs = sample_layer_hs[:, visual_indices, :]  # (n_layer, n_visual, d)

        # Cosine similarity → query importance weighting
        query_hs_norm = F.normalize(query_hs, dim=-1)
        visual_hs_norm = F.normalize(visual_hs, dim=-1)
        sim_matrix = torch.einsum('lqd,lvd->lqv', query_hs_norm, visual_hs_norm)
        attn_per_query = sim_matrix.sum(dim=-1)  # (n_layer, n_query)

        # Top-k query selection (aggregated across layers)
        k = min(self.query_topk, len(query_indices))
        agg_attn = attn_per_query.sum(dim=0)  # (n_query,)
        _, topk_local_idx = torch.topk(agg_attn, k, largest=True)

        # Aggregate attention: target token → visual tokens, weighted by query importance
        num_layers = len(calculated_attention[0])
        epsilon = 1e-8

        all_head_attns = []
        query_head_attns = []

        for layer_idx in range(num_layers):
            layer_attn = calculated_attention[0][layer_idx][0]  # (H, Q+1, seq_len)
            # Target attention (last merged index = target_idx)
            target_attn = layer_attn[:, -1, visual_indices]  # (H, n_visual)
            all_head_attns.append(target_attn)

            # Query attention for weighting
            q_attn = layer_attn[:, topk_local_idx, :][:, :, visual_indices]  # (H, k, n_visual)
            query_head_attns.append(q_attn)

        # Concatenate across layers: (n_layers * H, n_visual) and (n_layers * H, k, n_visual)
        all_head_attns_cat = torch.cat(all_head_attns, dim=0)
        query_head_attns_cat = torch.cat(query_head_attns, dim=0)

        # Head weights from query attention
        head_weights = query_head_attns_cat.sum(dim=(-1, -2))  # (n_layers * H,)
        head_weights = head_weights.softmax(dim=-1)

        # Weighted aggregation
        attn_merged = torch.mul(head_weights[:, None], all_head_attns_cat)  # (LH, n_visual)
        attn_merged = attn_merged.sum(dim=0, keepdim=True)  # (1, n_visual)
        attn_merged = attn_merged / (attn_merged.sum(dim=-1, keepdim=True) + epsilon)

        # Get spatial dimensions
        merge_size = self.model.config.vision_config.spatial_merge_size
        _, n_height, n_width = (inputs["image_grid_thw"][0] // merge_size).tolist()
        n_height, n_width = int(n_height), int(n_width)

        # Region-based prediction
        pred_x, pred_y = get_prediction_from_attention(
            attn_merged, n_width, n_height
        )

        # Map crop-local coords to original image coords
        x1, y1, x2, y2 = crop_bbox
        orig_x = x1 + pred_x * (x2 - x1)
        orig_y = y1 + pred_y * (y2 - y1)

        debug = {
            "text": generated_text,
            "n_visual": n_visual,
            "n_width": n_width,
            "n_height": n_height,
            "crop_pred": (pred_x, pred_y),
            "orig_pred": (orig_x, orig_y),
            "attn_max": attn_merged.max().item(),
            "attn_min": attn_merged.min().item(),
        }

        return orig_x, orig_y, debug

    @torch.no_grad()
    def predict(
        self,
        image,
        instruction: str,
        fixation: Optional[tuple[float, float]] = None,
        num_rounds: int = 3,
        max_new_tokens: int = 50,
    ) -> tuple[float, float]:
        """Iterative foveated inference: progressively zoom in on the target.

        Round 1: periphery (full image, low-res) → find attention peak
        Round 2: parafovea crop around peak → refine peak
        Round 3: fovea crop around refined peak → final prediction

        Args:
            image: PIL Image or path to screenshot.
            instruction: Text instruction for grounding.
            fixation: Optional initial fixation (overrides Round 1 attention).
            num_rounds: Number of refinement rounds (1-3).
            max_new_tokens: Tokens to generate per round.

        Returns:
            (x, y) normalized coordinates of predicted click point.
        """
        from PIL import Image as PILImage

        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")

        W, H = image.size

        # === Round 1: Periphery (full image, low resolution) ===
        periphery_h = int(self.sampler.periphery_resolution * H / W)
        periphery = image.resize(
            (self.sampler.periphery_resolution, periphery_h),
            PILImage.LANCZOS,
        )

        fx, fy, debug1 = self._single_round_inference(
            periphery, instruction,
            crop_bbox=(0.0, 0.0, 1.0, 1.0),
            max_new_tokens=max_new_tokens,
        )
        print(f"  Round 1 (periphery): pred=({fx:.4f}, {fy:.4f}), "
              f"attn_range=[{debug1['attn_min']:.6f}, {debug1['attn_max']:.6f}], "
              f"text={debug1['text'][:80]}")

        if fixation is not None:
            fx, fy = fixation

        if num_rounds < 2:
            return (fx, fy)

        # === Round 2: Parafovea (medium crop around attention peak) ===
        para_crop = self.sampler._extract_crop(
            image, fx, fy, self.sampler.parafovea_size, self.sampler.parafovea_resolution
        )
        fx, fy, debug2 = self._single_round_inference(
            para_crop["image"], instruction,
            crop_bbox=para_crop["bbox"],
            max_new_tokens=max_new_tokens,
        )
        print(f"  Round 2 (parafovea): pred=({fx:.4f}, {fy:.4f}), "
              f"attn_range=[{debug2['attn_min']:.6f}, {debug2['attn_max']:.6f}], "
              f"text={debug2['text'][:80]}")

        if num_rounds < 3:
            return (fx, fy)

        # === Round 3: Fovea (high-res crop around refined peak) ===
        fovea_crop = self.sampler._extract_crop(
            image, fx, fy, self.sampler.fovea_size, self.sampler.fovea_resolution
        )
        fx, fy, debug3 = self._single_round_inference(
            fovea_crop["image"], instruction,
            crop_bbox=fovea_crop["bbox"],
            max_new_tokens=max_new_tokens,
        )
        print(f"  Round 3 (fovea): pred=({fx:.4f}, {fy:.4f}), "
              f"attn_range=[{debug3['attn_min']:.6f}, {debug3['attn_max']:.6f}], "
              f"text={debug3['text'][:80]}")

        return (fx, fy)
