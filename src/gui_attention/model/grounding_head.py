"""
Grounding Head for GUI-Attention.

Predicts (x, y) coordinates from anchor token attention over visual tokens.
Based on GUI-AIMA's GroundingHead_MultiPatch_Attention with foveated adaptations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FoveatedGroundingHead(nn.Module):
    """Grounding head that operates on foveated visual representations.

    Takes the anchor token's attention distribution over multi-resolution
    visual tokens and predicts click coordinates.

    Args:
        hidden_size: Hidden dimension of the language model.
        num_heads: Number of attention heads to aggregate.
        num_layers: Number of layers to aggregate attention from.
        fovea_levels: Number of foveation resolution levels.
    """

    def __init__(
        self,
        hidden_size: int = 1536,  # Qwen2.5-VL-3B
        num_heads: int = 16,
        num_layers: int = 36,
        fovea_levels: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.fovea_levels = fovea_levels

        # Per-head, per-layer learnable weights (like GUI-AIMA's visual-sink)
        self.head_weights = nn.Parameter(torch.ones(num_layers, num_heads))
        # Level-wise calibration for multi-resolution tokens
        self.level_scale = nn.Parameter(torch.ones(fovea_levels))

    def forward(
        self,
        anchor_attention: torch.Tensor,
        visual_token_coords: torch.Tensor,
        fovea_level_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict (x, y) from anchor attention over visual tokens.

        Args:
            anchor_attention: (B, num_layers, num_heads, num_visual_tokens)
                Attention from anchor token to all visual tokens.
            visual_token_coords: (B, num_visual_tokens, 2)
                Normalized (x, y) center coordinate for each visual token.
            fovea_level_ids: (B, num_visual_tokens)
                Which foveation level each token belongs to (0=fovea, 1=parafovea, ...).

        Returns:
            coords: (B, 2) predicted normalized (x, y) coordinates.
        """
        B = anchor_attention.shape[0]

        # Weighted combination across layers and heads
        weights = F.softmax(self.head_weights.view(-1), dim=0)  # (num_layers * num_heads,)
        attn = rearrange(anchor_attention, "b l h v -> b (l h) v")  # (B, L*H, V)
        attn = torch.einsum("blv,l->bv", attn, weights)  # (B, V)

        # Apply level-wise calibration
        level_scales = self.level_scale[fovea_level_ids]  # (B, V)
        attn = attn * level_scales

        # Normalize to probability distribution
        attn = F.softmax(attn, dim=-1)  # (B, V)

        # Weighted sum of coordinates
        coords = torch.einsum("bv,bvc->bc", attn, visual_token_coords)  # (B, 2)

        return coords
