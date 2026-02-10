"""
Foveated Visual Token Sampler.

Implements human-vision-inspired multi-resolution visual token sampling
for GUI screenshots. Produces a single set of visual tokens with:
  - High resolution (fovea) at the attention center
  - Medium resolution (parafovea) in the surrounding region
  - Low resolution (periphery) for the rest of the screen

This replaces GUI-AIMA's 2-step zoom-in with a single forward pass.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image


@dataclass
class FoveatedTokens:
    """Container for foveated visual token information.

    Attributes:
        pixel_values: Preprocessed image tensor ready for the vision encoder.
        token_coords: (N, 2) normalized (x, y) center of each visual token in original image space.
        level_ids: (N,) foveation level index for each token (0=fovea, 1=parafovea, 2=periphery).
        num_tokens_per_level: List of token counts per level.
    """
    pixel_values: torch.Tensor
    token_coords: torch.Tensor
    level_ids: torch.Tensor
    num_tokens_per_level: list[int]


class FoveatedSampler:
    """Generate foveated multi-resolution visual tokens from a screenshot.

    The sampler creates a composite image or token set that mimics human
    foveated vision: high acuity at the fixation point, rapidly decreasing
    resolution in the periphery.

    Strategy options:
        1. **crop-and-resize**: Extract regions at different scales, resize each
           to fixed token budget, concatenate. Simple, works with any ViT.
        2. **adaptive-tiling**: Use Qwen2.5-VL's native dynamic resolution but
           allocate more tiles to the foveal region.
        3. **token-merge**: Full-resolution encoding followed by progressive
           token merging in peripheral regions.

    We start with strategy 1 (crop-and-resize) for simplicity and compatibility.

    Args:
        fovea_size: Size of the foveal region as fraction of image dimension.
        parafovea_size: Size of the parafoveal region as fraction.
        fovea_resolution: Target resolution (pixels) for foveal crop.
        parafovea_resolution: Target resolution for parafoveal crop.
        periphery_resolution: Target resolution for full image (periphery).
        target_total_tokens: Approximate total visual token budget.
    """

    def __init__(
        self,
        fovea_size: float = 0.15,
        parafovea_size: float = 0.4,
        fovea_resolution: int = 768,
        parafovea_resolution: int = 512,
        periphery_resolution: int = 384,
        target_total_tokens: int = 5000,
    ):
        self.fovea_size = fovea_size
        self.parafovea_size = parafovea_size
        self.fovea_resolution = fovea_resolution
        self.parafovea_resolution = parafovea_resolution
        self.periphery_resolution = periphery_resolution
        self.target_total_tokens = target_total_tokens

    def sample(
        self,
        image: Image.Image,
        fixation: Optional[tuple[float, float]] = None,
    ) -> dict:
        """Generate foveated crops from an image.

        Args:
            image: PIL Image (screenshot).
            fixation: Optional (x, y) normalized fixation point [0, 1].
                      If None, uses image center (first pass) or predicted
                      attention center (iterative refinement).

        Returns:
            Dict with crops and metadata for each foveation level.
        """
        W, H = image.size

        if fixation is None:
            fx, fy = 0.5, 0.5
        else:
            fx, fy = fixation

        # Compute crop regions
        crops = []

        # Level 0: Fovea (highest resolution, smallest region)
        fovea_crop = self._extract_crop(image, fx, fy, self.fovea_size, self.fovea_resolution)
        crops.append({"level": 0, "name": "fovea", **fovea_crop})

        # Level 1: Parafovea (medium resolution, medium region)
        para_crop = self._extract_crop(image, fx, fy, self.parafovea_size, self.parafovea_resolution)
        crops.append({"level": 1, "name": "parafovea", **para_crop})

        # Level 2: Periphery (low resolution, full image)
        periphery = image.resize(
            (self.periphery_resolution, int(self.periphery_resolution * H / W)),
            Image.LANCZOS,
        )
        crops.append({
            "level": 2,
            "name": "periphery",
            "image": periphery,
            "bbox": (0.0, 0.0, 1.0, 1.0),
        })

        return {
            "crops": crops,
            "fixation": (fx, fy),
            "original_size": (W, H),
        }

    def _extract_crop(
        self,
        image: Image.Image,
        cx: float,
        cy: float,
        region_size: float,
        target_res: int,
    ) -> dict:
        """Extract and resize a crop centered at (cx, cy).

        Args:
            image: Source image.
            cx, cy: Normalized center coordinates.
            region_size: Fraction of image dimension for crop.
            target_res: Target resolution to resize crop to.

        Returns:
            Dict with cropped image and normalized bbox.
        """
        W, H = image.size
        half_w = region_size * W / 2
        half_h = region_size * H / 2

        px, py = cx * W, cy * H

        x1 = max(0, int(px - half_w))
        y1 = max(0, int(py - half_h))
        x2 = min(W, int(px + half_w))
        y2 = min(H, int(py + half_h))

        crop = image.crop((x1, y1, x2, y2))
        crop_resized = crop.resize((target_res, int(target_res * (y2 - y1) / max(x2 - x1, 1))), Image.LANCZOS)

        bbox = (x1 / W, y1 / H, x2 / W, y2 / H)

        return {"image": crop_resized, "bbox": bbox}
