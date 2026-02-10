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
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

        # Foveated sampler
        self.sampler = FoveatedSampler(**(fovea_config or {}))

        # Grounding head
        config = self.model.config
        self.grounding_head = FoveatedGroundingHead(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_layers=config.num_hidden_layers,
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
        # TODO: Implement inference pipeline
        # 1. Foveated sampling
        # 2. Process crops through Qwen2.5-VL
        # 3. Extract anchor attention
        # 4. Grounding head prediction
        raise NotImplementedError("Inference pipeline - to be implemented")
