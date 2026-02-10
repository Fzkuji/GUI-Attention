"""
Foveated Qwen2.5-VL for GUI Grounding.

Extends Qwen2.5-VL with:
1. Foveated visual input (multi-resolution crops → single token sequence)
2. Anchor-based grounding head (adapted from GUI-AIMA)
3. Efficient single-pass inference

The key insight: instead of encoding one ultra-high-res image (18K tokens)
or doing 2-step zoom-in, we encode 3 crops at different resolutions
(~5K tokens total) in a single forward pass.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor

from gui_attention.foveation.sampler import FoveatedSampler
from gui_attention.model.grounding_head import FoveatedGroundingHead


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

    Args:
        model_name_or_path: Pretrained Qwen2.5-VL checkpoint.
        fovea_config: Dict of FoveatedSampler parameters.
        grounding_config: Dict of FoveatedGroundingHead parameters.
        anchor_token: Special token for grounding anchor.
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
        # Use "eager" attention to allow output_attentions=True for grounding head
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

        # Freeze base model, train only grounding head + level scales
        self._freeze_base_model()

    def _freeze_base_model(self):
        """Freeze base Qwen2.5-VL parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze grounding head
        for param in self.grounding_head.parameters():
            param.requires_grad = True

    def forward(self, **kwargs):
        """Training forward pass.

        TODO: Implement training logic with:
        - Multi-resolution image encoding
        - Anchor attention extraction
        - Grounding head prediction
        - L1 + smooth-L1 loss on coordinates
        """
        raise NotImplementedError("Training forward pass - to be implemented")

    def _build_foveated_messages(self, crops: list[dict], instruction: str) -> list[dict]:
        """Build Qwen2.5-VL chat messages with foveated crops as multi-image input.

        Each crop becomes a separate <image> in the conversation, labeled by level.
        """
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
        self, crops: list[dict], input_ids: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Compute normalized (x,y) coords and level IDs for each visual token.

        Returns:
            token_coords: (num_visual_tokens, 2) in original image space [0,1].
            level_ids: (num_visual_tokens,) level index per token.
            visual_mask: Boolean mask over input_ids indicating visual tokens.
        """
        # Qwen2.5-VL uses token id 151655 for <|image_pad|>
        IMAGE_PAD_ID = 151655
        visual_mask = (input_ids == IMAGE_PAD_ID)

        all_coords = []
        all_levels = []

        for i, crop in enumerate(crops):
            bbox = crop["bbox"]  # (x1_n, y1_n, x2_n, y2_n) normalized
            x1, y1, x2, y2 = bbox

            # Number of visual tokens for this crop from image_grid_thw
            # Qwen2.5-VL applies spatial_merge (2x2) so actual tokens = t * (h/m) * (w/m)
            merge = self.config.vision_config.spatial_merge_size  # typically 2
            if i < image_grid_thw.shape[0]:
                t, h, w = image_grid_thw[i].tolist()
                h_m, w_m = int(h) // merge, int(w) // merge
                n_tokens = int(t) * h_m * w_m
            else:
                n_tokens = 0
                h_m = w_m = 0

            # Generate grid coords on the MERGED grid (post spatial-merge)
            for rep in range(int(t) if n_tokens > 0 else 0):
                for row in range(h_m):
                    for col in range(w_m):
                        # Token center in crop-local normalized coords
                        lx = (col + 0.5) / w_m
                        ly = (row + 0.5) / h_m
                        # Map to original image coords
                        ox = x1 + lx * (x2 - x1)
                        oy = y1 + ly * (y2 - y1)
                        all_coords.append([ox, oy])
                        all_levels.append(crop["level"])

        device = input_ids.device
        token_coords = torch.tensor(all_coords, dtype=torch.float32, device=device)
        level_ids = torch.tensor(all_levels, dtype=torch.long, device=device)
        return token_coords, level_ids, visual_mask

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

        # Step 2: Build messages and process with Qwen2.5-VL processor
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
        # Temporarily switch to eager attention for extraction
        outputs = self.model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

        # Step 4: Extract anchor token attention over visual tokens
        input_ids = inputs["input_ids"][0]

        # Find anchor token position
        # The ANCHOR_TOKEN gets tokenized — find its position
        anchor_text_ids = self.processor.tokenizer.encode(self.ANCHOR_TOKEN, add_special_tokens=False)
        anchor_pos = None
        ids_list = input_ids.tolist()
        for pos in range(len(ids_list) - len(anchor_text_ids) + 1):
            if ids_list[pos:pos + len(anchor_text_ids)] == anchor_text_ids:
                anchor_pos = pos + len(anchor_text_ids) - 1  # last token of anchor
                break

        if anchor_pos is None:
            print("Warning: ANCHOR token not found, using last token position")
            anchor_pos = len(ids_list) - 1

        # Compute visual token coords
        token_coords, level_ids, visual_mask = self._compute_visual_token_coords(
            crops, input_ids, inputs.get("image_grid_thw", torch.zeros(0, 3))
        )

        visual_indices = visual_mask.nonzero(as_tuple=True)[0]

        # Extract attention: (num_layers, num_heads, seq_len, seq_len)
        # We want attention from anchor_pos to visual token positions
        num_layers = len(outputs.attentions)
        num_heads = outputs.attentions[0].shape[1]
        n_visual = visual_indices.shape[0]

        anchor_attn = torch.zeros(1, num_layers, num_heads, n_visual, device=self.model.device)
        for layer_idx, attn in enumerate(outputs.attentions):
            # attn shape: (B, num_heads, seq_len, seq_len)
            anchor_attn[0, layer_idx] = attn[0, :, anchor_pos, visual_indices]

        # Ensure coord counts match
        n_coords = token_coords.shape[0]
        if n_coords != n_visual:
            print(f"Warning: coord count ({n_coords}) != visual token count ({n_visual}), truncating/padding")
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
