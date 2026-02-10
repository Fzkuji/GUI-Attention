"""
Foveated Qwen2.5-VL for GUI Grounding.

Extends Qwen2.5-VL with:
1. Foveated visual input (multi-resolution crops → single token sequence)
2. Anchor-based grounding head (adapted from GUI-AIMA)
3. Efficient single-pass inference
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

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

    @torch.no_grad()
    def predict(
        self,
        image,
        instruction: str,
        fixation: Optional[tuple[float, float]] = None,
    ) -> tuple[float, float]:
        """Single-pass inference: screenshot + instruction → (x, y).

        Args:
            image: PIL Image or path to screenshot.
            instruction: Text instruction (e.g., "Click the search button").
            fixation: Optional initial fixation point. If None, uses center.

        Returns:
            (x, y) normalized coordinates of predicted click point.
        """
        from PIL import Image as PILImage

        if isinstance(image, str):
            image = PILImage.open(image).convert("RGB")

        # Step 1: Foveated sampling
        foveated = self.sampler.sample(image, fixation=fixation)
        crops = foveated["crops"]

        # Step 2: Build messages and process
        messages = self._build_foveated_messages(crops, instruction)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        crop_images = [c["image"] for c in crops]
        inputs = self.processor(
            text=[text],
            images=crop_images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Step 3: Forward pass with attention output
        outputs = self.model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

        # Step 4: Extract anchor attention over visual tokens
        input_ids = inputs["input_ids"][0]
        anchor_pos = self._find_anchor_position(input_ids)

        crop_bboxes = [c["bbox"] for c in crops]
        crop_levels = [c["level"] for c in crops]
        token_coords, level_ids, visual_mask = self._compute_visual_token_coords(
            crop_bboxes, crop_levels, input_ids,
            inputs.get("image_grid_thw", torch.zeros(0, 3))
        )

        visual_indices = visual_mask.nonzero(as_tuple=True)[0]
        anchor_attn = self._extract_anchor_attention(
            outputs.attentions, anchor_pos, visual_indices
        )

        # Align counts
        n_coords = token_coords.shape[0]
        n_visual = visual_indices.shape[0]
        if n_coords != n_visual:
            min_n = min(n_coords, n_visual)
            token_coords = token_coords[:min_n]
            level_ids = level_ids[:min_n]
            anchor_attn = anchor_attn[:, :, :, :min_n]

        # Step 5: Grounding head prediction
        coords = self.grounding_head(
            anchor_attention=anchor_attn,
            visual_token_coords=token_coords.unsqueeze(0),
            fovea_level_ids=level_ids.unsqueeze(0),
        )

        x, y = coords[0].tolist()
        return (x, y)
