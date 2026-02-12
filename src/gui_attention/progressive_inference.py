"""
Progressive Resolution Enhancement for GUI Grounding.

Core idea: In a single forward pass, progressively add higher-resolution
visual tokens for attended regions, using the temporal dimension of M-RoPE
to encode resolution levels.

- t=0: Low-res full image (first glance)
- t=1: Medium-res focused region (second look)
- t=2: High-res precise region (third look)

Spatial positions (h, w) remain consistent across levels,
so the model naturally understands the spatial relationship.
"""

import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


@dataclass
class ProgressiveConfig:
    """Configuration for progressive resolution enhancement."""
    # Level 1: Full image, low resolution
    l1_max_pixels: int = 56 * 56 * 28 * 28  # ~3000 tokens after merge
    
    # Level 2: Focused region, medium resolution
    l2_crop_ratio: float = 0.4  # crop 40% of image around fixation
    l2_max_pixels: int = 56 * 56 * 28 * 28
    
    # Level 3: Precise region, high resolution
    l3_crop_ratio: float = 0.15
    l3_max_pixels: int = 56 * 56 * 28 * 28
    
    # General
    min_pixels: int = 4 * 28 * 28
    num_levels: int = 3
    attention_threshold: float = 0.3  # threshold for selecting focus region
    

class ProgressiveInference:
    """
    Progressive Resolution Enhancement using M-RoPE temporal dimension.
    
    Key mechanism:
    1. Encode full image at low resolution (t=0)
    2. Run forward, extract attention → find focus region
    3. Encode focus region at higher resolution
    4. Append to sequence with t=1, same spatial coords
    5. Continue forward (reuse KV cache)
    6. Repeat for t=2 if needed
    """
    
    def __init__(
        self,
        model,
        processor,
        config: Optional[ProgressiveConfig] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.processor = processor
        self.config = config or ProgressiveConfig()
        self.device = device
        
        # Pre-create processors for each level
        self.level_processors = {}
        for level, max_px in enumerate([
            self.config.l1_max_pixels,
            self.config.l2_max_pixels, 
            self.config.l3_max_pixels,
        ]):
            self.level_processors[level] = AutoProcessor.from_pretrained(
                model.config._name_or_path,
                min_pixels=self.config.min_pixels,
                max_pixels=max_px,
            )
            # Share tokenizer
            self.level_processors[level].tokenizer = processor.tokenizer
    
    def _get_visual_token_grid(self, image_grid_thw):
        """Get the spatial dimensions of visual tokens after merge."""
        merge_size = self.model.visual.spatial_merge_size
        t, h, w = (image_grid_thw[0] // merge_size).tolist()
        return int(t), int(h), int(w)
    
    def _crop_region(
        self, image: Image.Image, center_x: float, center_y: float, crop_ratio: float
    ) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
        """Crop a region around (center_x, center_y)."""
        W, H = image.size
        crop_w = int(W * crop_ratio)
        crop_h = int(H * crop_ratio)
        
        cx_px = int(center_x * W)
        cy_px = int(center_y * H)
        
        x1 = max(0, cx_px - crop_w // 2)
        y1 = max(0, cy_px - crop_h // 2)
        x2 = min(W, x1 + crop_w)
        y2 = min(H, y1 + crop_h)
        
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
        
        cropped = image.crop((x1, y1, x2, y2))
        bbox = (x1 / W, y1 / H, x2 / W, y2 / H)
        return cropped, bbox
    
    def _encode_image_at_level(
        self, image: Image.Image, level: int
    ) -> Dict:
        """
        Encode an image through ViT at a specific resolution level.
        Returns visual tokens and their grid dimensions.
        """
        proc = self.level_processors[level]
        
        # Process image to get pixel values
        # We need to go through the processor to get properly resized image
        dummy_text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n<|im_end|>"
        inputs = proc(
            text=[dummy_text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)
        
        # Extract just the visual encoding
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        
        # Run through ViT
        with torch.no_grad():
            visual_tokens = self.model.visual(
                pixel_values, grid_thw=image_grid_thw
            )
        
        _, n_h, n_w = self._get_visual_token_grid(image_grid_thw)
        
        return {
            "tokens": visual_tokens,  # (1, n_tokens, hidden_dim)
            "n_height": n_h,
            "n_width": n_w,
            "n_tokens": visual_tokens.shape[1],
            "image_grid_thw": image_grid_thw,
        }
    
    def _compute_position_ids_for_level(
        self,
        level: int,
        n_height: int,
        n_width: int,
        crop_bbox: Tuple[float, float, float, float],
        l1_n_height: int,
        l1_n_width: int,
        seq_offset: int,
    ) -> torch.Tensor:
        """
        Compute M-RoPE position_ids (3, n_tokens) for visual tokens at a given level.
        
        Key: temporal dimension = level, spatial dimensions map to L1's coordinate system.
        
        Args:
            level: resolution level (0, 1, 2)
            n_height, n_width: grid dimensions of this level's tokens
            crop_bbox: (x1, y1, x2, y2) in normalized coords of the crop region
            l1_n_height, l1_n_width: L1's grid dimensions (reference frame)
            seq_offset: position offset in the full sequence
        """
        n_tokens = n_height * n_width
        
        # Temporal dimension: level index
        t_ids = torch.full((n_tokens,), level, dtype=torch.long, device=self.device)
        
        # Spatial dimensions: map to L1's coordinate space
        bx1, by1, bx2, by2 = crop_bbox
        
        h_ids = torch.zeros(n_tokens, dtype=torch.long, device=self.device)
        w_ids = torch.zeros(n_tokens, dtype=torch.long, device=self.device)
        
        for i in range(n_tokens):
            row = i // n_width
            col = i % n_width
            
            # Normalized position within this crop [0, 1]
            norm_h = (row + 0.5) / n_height
            norm_w = (col + 0.5) / n_width
            
            # Map to position in original image [0, 1]
            orig_h = by1 + norm_h * (by2 - by1)
            orig_w = bx1 + norm_w * (bx2 - bx1)
            
            # Map to L1's grid coordinates
            h_in_l1 = orig_h * l1_n_height
            w_in_l1 = orig_w * l1_n_width
            
            # Use integer positions (round to nearest L1 grid cell)
            # This ensures compatibility with trained position embeddings
            h_ids[i] = int(round(h_in_l1))
            w_ids[i] = int(round(w_in_l1))
        
        # Stack as (3, n_tokens): [temporal, height, width]
        position_ids = torch.stack([t_ids, h_ids, w_ids], dim=0)
        
        return position_ids
    
    def _extract_attention_focus(
        self,
        attention_scores,
        n_height: int,
        n_width: int,
    ) -> Tuple[float, float, float]:
        """
        Extract the focus point from attention scores.
        Returns (center_x, center_y, max_attention).
        """
        # Reshape to 2D grid
        attn_2d = attention_scores[:n_height * n_width].view(n_height, n_width)
        
        # Weighted average of positions
        weights = F.softmax(attn_2d.flatten(), dim=0).view(n_height, n_width)
        
        h_coords = torch.arange(n_height, dtype=torch.float, device=weights.device)
        w_coords = torch.arange(n_width, dtype=torch.float, device=weights.device)
        
        center_h = (weights.sum(dim=1) * h_coords).sum() / n_height
        center_w = (weights.sum(dim=0) * w_coords).sum() / n_width
        
        attn_max = attn_2d.max().item()
        
        return center_w.item(), center_h.item(), attn_max
    
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        instruction: str,
        verbose: bool = False,
    ) -> Tuple[float, float]:
        """
        Progressive resolution inference.
        
        This is the main entry point. Currently implemented as a conceptual
        prototype — the full KV-cache-based incremental approach requires
        deeper integration with the model's generate/forward methods.
        
        For now, we simulate the progressive approach by:
        1. Building the full multi-level token sequence
        2. Running a single forward pass with custom position_ids
        3. Extracting attention over all visual tokens
        
        TODO: Implement true incremental forward with KV cache reuse.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # === Level 1: Low-res full image ===
        l1 = self._encode_image_at_level(image, level=0)
        l1_pos = self._compute_position_ids_for_level(
            level=0,
            n_height=l1["n_height"],
            n_width=l1["n_width"],
            crop_bbox=(0.0, 0.0, 1.0, 1.0),
            l1_n_height=l1["n_height"],
            l1_n_width=l1["n_width"],
            seq_offset=0,
        )
        
        if verbose:
            print(f"  L1: {l1['n_tokens']} tokens ({l1['n_height']}x{l1['n_width']})")
        
        # For the prototype, we need to get L1 attention first to determine
        # where to focus for L2. We'll do this with a quick forward pass.
        # In the final version, this would be part of the incremental forward.
        
        # ... (placeholder for attention extraction)
        # For now, return a simple prediction based on L1 only
        
        # TODO: Implement full progressive pipeline
        # 1. L1 forward → attention → focus point
        # 2. Crop focus region → L2 encode
        # 3. Append L2 tokens with t=1 position_ids
        # 4. Incremental forward → refined attention
        # 5. Optional: L3
        # 6. Final grounding from multi-level attention
        
        return 0.5, 0.5  # Placeholder


def demo_position_encoding():
    """
    Demonstrate how M-RoPE position_ids work for progressive resolution.
    This can run without GPU to verify the concept.
    """
    print("=== Progressive Resolution M-RoPE Demo ===\n")
    
    # Simulate a 4x4 L1 grid (16 tokens)
    l1_h, l1_w = 4, 4
    
    print("Level 1 (full image, low-res):")
    print(f"  Grid: {l1_h}x{l1_w} = {l1_h*l1_w} tokens")
    print(f"  Positions (t, h, w):")
    for r in range(l1_h):
        for c in range(l1_w):
            print(f"    token[{r*l1_w+c:2d}]: t=0, h={r}, w={c}")
    
    # L2: Focus on region (0.25, 0.25) to (0.75, 0.75) → L1 cells (1,1)-(2,2)
    # Encoded at 2x resolution → 4x4 grid for that region
    l2_h, l2_w = 4, 4
    crop_bbox = (0.25, 0.25, 0.75, 0.75)
    
    print(f"\nLevel 2 (focused region, medium-res):")
    print(f"  Crop: {crop_bbox}")
    print(f"  Grid: {l2_h}x{l2_w} = {l2_h*l2_w} tokens")
    print(f"  Positions (t, h, w):")
    for r in range(l2_h):
        for c in range(l2_w):
            norm_h = (r + 0.5) / l2_h
            norm_w = (c + 0.5) / l2_w
            orig_h = crop_bbox[1] + norm_h * (crop_bbox[3] - crop_bbox[1])
            orig_w = crop_bbox[0] + norm_w * (crop_bbox[2] - crop_bbox[0])
            h_in_l1 = orig_h * l1_h
            w_in_l1 = orig_w * l1_w
            print(f"    token[{r*l2_w+c:2d}]: t=1, h={round(h_in_l1)}, w={round(w_in_l1)}")
    
    print(f"\nKey insight:")
    print(f"  L1 token at (t=0, h=1, w=1) covers the top-left of the focus region")
    print(f"  L2 tokens at (t=1, h=1, w=1) provide fine details for the SAME spatial location")
    print(f"  The model sees both through attention — temporal dim tells it L2 is a refinement")
    
    # Total tokens
    total = l1_h * l1_w + l2_h * l2_w
    full_res = (l1_h * 2) * (l1_w * 2)  # if entire image at L2 resolution
    print(f"\n  Total tokens: {total} (progressive) vs {full_res} (full L2 resolution)")
    print(f"  Savings: {(1 - total/full_res)*100:.0f}%")


if __name__ == "__main__":
    demo_position_encoding()
