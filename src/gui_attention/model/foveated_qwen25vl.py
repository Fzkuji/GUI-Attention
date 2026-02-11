"""
Foveated Qwen2.5-VL for GUI Grounding.

Extends Qwen2.5-VL with:
1. Foveated visual input (multi-resolution crops → single token sequence)
2. Anchor-based grounding head (adapted from GUI-AIMA)
3. Generation-based attention extraction for semantic grounding
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
from gui_attention.model.grounding_head import FoveatedGroundingHead


# Qwen2.5-VL uses token id 151655 for <|image_pad|>
IMAGE_PAD_ID = 151655


class FoveatedQwen25VL(nn.Module):
    """Qwen2.5-VL with foveated visual attention for GUI grounding.

    Architecture:
        Screenshot → FoveatedSampler → [fovea, parafovea, periphery crops]
                                         ↓
        Qwen2.5-VL vision encoder → visual tokens (with level IDs)
                                         ↓
        Qwen2.5-VL LM (with <ANCHOR> token) → anchor attention extraction
                                         ↓
        FoveatedGroundingHead → (x, y) coordinates
    """

    ANCHOR_TOKEN = "<ANCHOR>"

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        fovea_config: Optional[dict] = None,
        grounding_config: Optional[dict] = None,
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

        # Grounding head
        text_config = self.model.config.text_config
        self.grounding_head = FoveatedGroundingHead(
            hidden_size=text_config.hidden_size,
            num_heads=text_config.num_attention_heads,
            num_layers=text_config.num_hidden_layers,
            **(grounding_config or {}),
        )

        # Freeze base model, train only grounding head
        self._freeze_base_model()

    def _freeze_base_model(self):
        """Freeze base Qwen2.5-VL parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.grounding_head.parameters():
            param.requires_grad = True

    def _build_foveated_messages(self, crops: list[dict], instruction: str) -> list[dict]:
        """Build Qwen2.5-VL chat messages with foveated crops as multi-image input."""
        image_content = []
        for crop in crops:
            image_content.append({"type": "image", "image": crop["image"]})
        image_content.append({
            "type": "text",
            "text": (
                f"The above images show a GUI screenshot at three resolutions: "
                f"fovea (high-res center), parafovea (medium-res), and periphery (full screen low-res). "
                f"Instruction: {instruction}\n{self.ANCHOR_TOKEN}"
            ),
        })
        return [{"role": "user", "content": image_content}]

    def _compute_visual_token_coords(
        self,
        crop_bboxes: list[tuple],
        crop_levels: list[int],
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute normalized (x,y) coords and level IDs for each visual token.

        Each visual token maps to a specific spatial position in the original image.
        The mapping accounts for:
        - Which crop the token belongs to (determined by image_grid_thw order)
        - The crop's bbox in original image space
        - Spatial merging (2x2 by default in Qwen2.5-VL)

        Args:
            crop_bboxes: List of (x1, y1, x2, y2) normalized bboxes per crop.
            crop_levels: List of level indices per crop (0=fovea, 1=para, 2=periphery).
            input_ids: Token IDs for the sequence.
            image_grid_thw: (num_images, 3) tensor of (t, h, w) per image.

        Returns:
            token_coords: (num_visual_tokens, 2) in original image space [0,1].
            level_ids: (num_visual_tokens,) level index per token.
            visual_mask: Boolean mask over input_ids indicating visual tokens.
        """
        visual_mask = (input_ids == IMAGE_PAD_ID)
        merge = self.model.config.vision_config.spatial_merge_size  # typically 2

        all_coords = []
        all_levels = []

        for i in range(min(len(crop_bboxes), image_grid_thw.shape[0])):
            bbox = crop_bboxes[i]
            level = crop_levels[i]
            x1, y1, x2, y2 = bbox

            t, h, w = image_grid_thw[i].tolist()
            t, h, w = int(t), int(h), int(w)
            # After spatial merge, token grid is (h/merge, w/merge)
            h_m, w_m = h // merge, w // merge

            for _t in range(t):
                for row in range(h_m):
                    for col in range(w_m):
                        # Token center in crop-local normalized coords
                        lx = (col + 0.5) / w_m
                        ly = (row + 0.5) / h_m
                        # Map to original image coords via bbox
                        ox = x1 + lx * (x2 - x1)
                        oy = y1 + ly * (y2 - y1)
                        all_coords.append([ox, oy])
                        all_levels.append(level)

        device = input_ids.device
        token_coords = torch.tensor(all_coords, dtype=torch.float32, device=device)
        level_ids = torch.tensor(all_levels, dtype=torch.long, device=device)
        return token_coords, level_ids, visual_mask

    def _find_anchor_position(self, input_ids: torch.Tensor) -> int:
        """Find the position of the ANCHOR token in input_ids."""
        anchor_text_ids = self.processor.tokenizer.encode(self.ANCHOR_TOKEN, add_special_tokens=False)
        ids_list = input_ids.tolist()
        for pos in range(len(ids_list) - len(anchor_text_ids) + 1):
            if ids_list[pos:pos + len(anchor_text_ids)] == anchor_text_ids:
                return pos + len(anchor_text_ids) - 1  # last token of anchor
        # Fallback: last non-pad token
        return len(ids_list) - 1

    def _extract_anchor_attention(
        self,
        attentions: tuple,
        anchor_pos: int,
        visual_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Extract attention from anchor token to visual tokens.

        Args:
            attentions: Tuple of (B, num_heads, seq_len, seq_len) per layer.
            anchor_pos: Position of anchor token.
            visual_indices: Indices of visual tokens in the sequence.

        Returns:
            anchor_attn: (1, num_layers, num_heads, num_visual_tokens)
        """
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        n_visual = visual_indices.shape[0]
        device = attentions[0].device

        anchor_attn = torch.zeros(1, num_layers, num_heads, n_visual, device=device)
        for layer_idx, attn in enumerate(attentions):
            # attn: (B, num_heads, seq_len, seq_len)
            anchor_attn[0, layer_idx] = attn[0, :, anchor_pos, visual_indices]

        return anchor_attn

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        gt_coords: torch.Tensor,
        crop_bboxes: list,
        crop_levels: list,
    ) -> dict:
        """Training forward pass.

        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
            pixel_values: Concatenated pixel values for all images in batch.
            image_grid_thw: (total_images, 3)
            gt_coords: (B, 2) ground truth normalized (x, y).
            crop_bboxes: List[List[tuple]] per sample.
            crop_levels: List[List[int]] per sample.

        Returns:
            dict with 'loss', 'pred_coords', 'gt_coords'.
        """
        B = input_ids.shape[0]
        device = input_ids.device

        # Forward through base model with attention output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_attentions=True,
            return_dict=True,
        )

        # Process each sample in the batch
        all_pred_coords = []
        num_crops_per_sample = 3  # fovea, parafovea, periphery

        for b in range(B):
            b_input_ids = input_ids[b]

            # Get crop metadata for this sample
            b_bboxes = crop_bboxes[b]
            b_levels = crop_levels[b]

            # Determine which image_grid_thw rows belong to this sample
            img_start = b * num_crops_per_sample
            img_end = img_start + num_crops_per_sample
            b_grid_thw = image_grid_thw[img_start:img_end]

            # Compute visual token coords
            token_coords, level_ids, visual_mask = self._compute_visual_token_coords(
                b_bboxes, b_levels, b_input_ids, b_grid_thw
            )
            visual_indices = visual_mask.nonzero(as_tuple=True)[0]

            # Find anchor position
            anchor_pos = self._find_anchor_position(b_input_ids)

            # Extract attention for this sample
            n_visual = visual_indices.shape[0]
            num_layers = len(outputs.attentions)
            num_heads = outputs.attentions[0].shape[1]

            anchor_attn = torch.zeros(1, num_layers, num_heads, n_visual, device=device)
            for layer_idx, attn in enumerate(outputs.attentions):
                anchor_attn[0, layer_idx] = attn[b, :, anchor_pos, visual_indices]

            # Align coord count with visual token count
            n_coords = token_coords.shape[0]
            if n_coords != n_visual:
                min_n = min(n_coords, n_visual)
                token_coords = token_coords[:min_n]
                level_ids = level_ids[:min_n]
                anchor_attn = anchor_attn[:, :, :, :min_n]

            # Grounding head prediction
            pred = self.grounding_head(
                anchor_attention=anchor_attn,
                visual_token_coords=token_coords.unsqueeze(0),
                fovea_level_ids=level_ids.unsqueeze(0),
            )
            all_pred_coords.append(pred.squeeze(0))

        pred_coords = torch.stack(all_pred_coords)  # (B, 2)

        # Loss: SmoothL1
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(pred_coords, gt_coords.to(device))

        return {
            "loss": loss,
            "pred_coords": pred_coords.detach(),
            "gt_coords": gt_coords,
        }

    def _single_image_forward(
        self,
        image: "Image.Image",
        instruction: str,
        max_new_tokens: int = 10,
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        """Forward a single image through Qwen2.5-VL and get generated text + attention.

        Args:
            image: Single PIL Image (one resolution level).
            instruction: Text instruction.
            max_new_tokens: Number of tokens to generate.

        Returns:
            (generated_text, attention_over_visual_tokens, visual_token_coords)
            - generated_text: str
            - attn_map: (num_visual_tokens,) aggregated attention weights
            - token_coords: (num_visual_tokens, 2) normalized coords in [0,1]
        """
        # Build message with single image
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"Instruction: {instruction}\nPlease identify the location of the target element."},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        # Generate with attention
        results = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_attentions=True,
            temperature=0.1,
        )

        # Decode generated text
        input_ids = inputs["input_ids"][0]
        generated_ids = results.sequences[0][len(input_ids):]
        generated_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Extract visual token indices
        visual_mask = (input_ids == IMAGE_PAD_ID)
        visual_indices = visual_mask.nonzero(as_tuple=True)[0]
        n_visual = visual_indices.shape[0]

        if n_visual == 0:
            return generated_text, torch.zeros(0), torch.zeros(0, 2)

        # Aggregate attention from GENERATION steps (not prefill!)
        # results.attentions[0] = prefill attentions (flat, useless)
        # results.attentions[t] for t>=1 = generation step t: tuple of layers, each (B, H, 1, seq_len_so_far)
        # Visual token indices are still valid in the KV cache positions
        num_gen_steps = len(results.attentions) - 1  # exclude prefill
        if num_gen_steps > 0:
            num_layers = len(results.attentions[1])
        else:
            num_layers = len(results.attentions[0])

        agg_attn = torch.zeros(n_visual, device=input_ids.device, dtype=torch.float32)

        if num_gen_steps > 0:
            # Use attention from all generation steps → visual tokens
            for step in range(1, len(results.attentions)):
                step_attns = results.attentions[step]  # tuple of layers
                for layer_attn in step_attns:
                    # layer_attn: (B, H, 1, current_seq_len)
                    # visual_indices are positions in original input, still valid
                    attn = layer_attn[0, :, 0, visual_indices]  # (H, n_visual)
                    agg_attn += attn.float().mean(dim=0)
            agg_attn /= (num_gen_steps * num_layers)
        else:
            # Fallback to prefill (shouldn't happen if max_new_tokens > 0)
            last_input_pos = len(input_ids) - 1
            for layer_attn in results.attentions[0]:
                attn = layer_attn[0, :, last_input_pos, visual_indices]
                agg_attn += attn.float().mean(dim=0)
            agg_attn /= num_layers

        # Compute visual token coordinates (full image = bbox (0,0,1,1))
        image_grid_thw = inputs.get("image_grid_thw", torch.zeros(0, 3))
        merge = self.model.config.vision_config.spatial_merge_size
        coords = []
        if image_grid_thw.shape[0] > 0:
            t, h, w = image_grid_thw[0].tolist()
            t, h, w = int(t), int(h), int(w)
            h_m, w_m = h // merge, w // merge
            for _t in range(t):
                for row in range(h_m):
                    for col in range(w_m):
                        coords.append([(col + 0.5) / w_m, (row + 0.5) / h_m])

        token_coords = torch.tensor(coords, dtype=torch.float32, device=input_ids.device)

        # Align
        min_n = min(len(token_coords), n_visual)
        token_coords = token_coords[:min_n]
        agg_attn = agg_attn[:min_n]

        return generated_text, agg_attn, token_coords

    def _find_attention_peak(
        self,
        attn_map: torch.Tensor,
        token_coords: torch.Tensor,
        bbox: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    ) -> tuple[float, float]:
        """Find the peak attention location in original image coordinates.

        Args:
            attn_map: (N,) attention weights over visual tokens.
            token_coords: (N, 2) coords in crop-local [0,1] space.
            bbox: (x1, y1, x2, y2) of the crop in original image space.

        Returns:
            (x, y) in original image normalized coordinates.
        """
        if len(attn_map) == 0:
            return 0.5, 0.5

        # Softmax to get distribution
        attn_probs = F.softmax(attn_map, dim=0)

        # Weighted average of coordinates (in crop space)
        cx = (attn_probs * token_coords[:, 0]).sum().item()
        cy = (attn_probs * token_coords[:, 1]).sum().item()

        # Map to original image space
        x1, y1, x2, y2 = bbox
        ox = x1 + cx * (x2 - x1)
        oy = y1 + cy * (y2 - y1)

        return ox, oy

    @torch.no_grad()
    def predict(
        self,
        image,
        instruction: str,
        fixation: Optional[tuple[float, float]] = None,
        num_rounds: int = 3,
        gen_tokens_per_round: int = 10,
    ) -> tuple[float, float]:
        """Iterative foveated inference: progressively zoom in on the target.

        Round 1: Input periphery (low-res full image) → generate → find attention peak
        Round 2: Crop parafovea around peak → generate → refine peak
        Round 3: Crop fovea around refined peak → generate → final coordinates

        Args:
            image: PIL Image or path to screenshot.
            instruction: Text instruction.
            fixation: Optional initial fixation (overrides Round 1 center).
            num_rounds: Number of refinement rounds (1-3).
            gen_tokens_per_round: Tokens to generate per round.

        Returns:
            (x, y) normalized coordinates of predicted click point.
        """
        from PIL import Image as PILImage

        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")

        W, H = image.size

        # Round 1: Periphery (full image, low resolution)
        periphery = image.resize(
            (self.sampler.periphery_resolution,
             int(self.sampler.periphery_resolution * H / W)),
            PILImage.LANCZOS,
        )
        gen_text, attn_map, token_coords = self._single_image_forward(
            periphery, instruction, max_new_tokens=gen_tokens_per_round
        )
        fx, fy = self._find_attention_peak(attn_map, token_coords, bbox=(0.0, 0.0, 1.0, 1.0))

        if num_rounds < 2:
            return (fx, fy)

        # Round 2: Parafovea (medium crop around attention peak)
        para_crop_data = self.sampler._extract_crop(
            image, fx, fy, self.sampler.parafovea_size, self.sampler.parafovea_resolution
        )
        gen_text2, attn_map2, token_coords2 = self._single_image_forward(
            para_crop_data["image"], instruction, max_new_tokens=gen_tokens_per_round
        )
        fx, fy = self._find_attention_peak(attn_map2, token_coords2, bbox=para_crop_data["bbox"])

        if num_rounds < 3:
            return (fx, fy)

        # Round 3: Fovea (high-res crop around refined attention peak)
        fovea_crop_data = self.sampler._extract_crop(
            image, fx, fy, self.sampler.fovea_size, self.sampler.fovea_resolution
        )
        gen_text3, attn_map3, token_coords3 = self._single_image_forward(
            fovea_crop_data["image"], instruction, max_new_tokens=gen_tokens_per_round
        )
        fx, fy = self._find_attention_peak(attn_map3, token_coords3, bbox=fovea_crop_data["bbox"])

        return (fx, fy)
