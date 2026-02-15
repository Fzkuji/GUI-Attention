"""MultiRoundInputBuilder: builds multi-round multi-precision conversation tokens.

Each round's image is pre-processed at its own resolution level so the processor
never re-sizes an image intended for a different precision level.

Tracks which precision level and spatial bbox each image belongs to, so that
after attention extraction we can identify which image an attended token is from.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize

from gui_attention.constants import (
    PLACEHOLDER_SUFFIX, precision_for_level,
    CHAT_TEMPLATE, GROUNDING_SYSTEM_MESSAGE,
)


@dataclass
class ImageInfo:
    """Metadata for one image in the multi-round context."""
    level: int                          # precision level (0-3)
    global_bbox: Tuple[float, ...]      # (x1, y1, x2, y2) in original image coords, normalised
    # Filled in after tokenisation:
    visual_range: Optional[Tuple[int, int]] = None  # (start, end) in input_ids


def _presize_image(image: Image.Image, max_pixels: int, min_pixels: int = 3136) -> Image.Image:
    """Pre-resize to the exact dims the Qwen2.5-VL processor would use."""
    w, h = image.size
    new_h, new_w = smart_resize(h, w, max_pixels=max_pixels, min_pixels=min_pixels)
    if (new_w, new_h) != (w, h):
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


class MultiRoundInputBuilder:
    """Incrementally builds multi-round conversation tokens with precision tracking."""

    # Large enough that the processor won't further resize pre-resized images.
    _PASSTHROUGH_MAX = 20_000_000

    def __init__(self, model_path: str, tokenizer, min_pixels: int = 3136):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.min_pixels = min_pixels
        self._processor_cache: Dict[int, AutoProcessor] = {}
        self._resized_images: List[Image.Image] = []
        self.image_infos: List[ImageInfo] = []

    def _get_processor(self, max_pixels: int):
        if max_pixels not in self._processor_cache:
            p = AutoProcessor.from_pretrained(
                self.model_path, min_pixels=self.min_pixels, max_pixels=max_pixels,
            )
            p.tokenizer = self.tokenizer
            self._processor_cache[max_pixels] = p
        return self._processor_cache[max_pixels]

    def reset(self):
        """Clear state for a new sample."""
        self._resized_images = []
        self.image_infos = []

    # ----- round 0 --------------------------------------------------------

    def build_round0(self, image_or_path, instruction: str, level: int = 0):
        """Build round-0 inputs (full image at given precision level).

        Returns (inputs_dict, raw_text, images_list).
        """
        if isinstance(image_or_path, dict):
            image_or_path = image_or_path["image_path"]

        max_px = precision_for_level(level)
        conv = [
            {"role": "system", "content": [{"type": "text", "text": GROUNDING_SYSTEM_MESSAGE}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_or_path},
                {"type": "text", "text": instruction},
            ]},
        ]
        text = self._get_processor(max_px).apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False, chat_template=CHAT_TEMPLATE,
        )
        text += PLACEHOLDER_SUFFIX
        images, _ = process_vision_info(conv)

        resized_r0 = _presize_image(images[0], max_px, self.min_pixels)
        self._resized_images = [resized_r0]
        self.image_infos = [ImageInfo(level=level, global_bbox=(0.0, 0.0, 1.0, 1.0))]

        inputs = self._get_processor(max_px)(
            text=[text], images=[resized_r0], return_tensors="pt", padding=True,
        )
        # Fill visual_range for round 0
        self._update_visual_ranges(inputs["input_ids"][0])
        return inputs, text, images

    # ----- subsequent rounds ----------------------------------------------

    def extend_with_crop(self, prev_text: str, prev_images: list,
                         crop_pil: Image.Image, crop_bbox: tuple,
                         level: int):
        """Append a crop at the specified precision level.

        Args:
            crop_bbox: (x1, y1, x2, y2) normalised in original image coords.
            level: precision level for this crop (0-3).

        Returns (inputs_dict, raw_text, images_list).
        """
        max_px = precision_for_level(level)
        round_num = len(self.image_infos)

        zoom_text = (
            f"\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"[Zoomed region round {round_num + 1} around "
            f"({crop_bbox[0]:.2f},{crop_bbox[1]:.2f})-({crop_bbox[2]:.2f},{crop_bbox[3]:.2f})]"
            f"<|im_end|>\n"
            + PLACEHOLDER_SUFFIX
        )
        new_text = prev_text + zoom_text
        new_images = prev_images + [crop_pil]

        resized_crop = _presize_image(crop_pil, max_px, self.min_pixels)
        self._resized_images.append(resized_crop)
        self.image_infos.append(ImageInfo(level=level, global_bbox=crop_bbox))

        proc = self._get_processor(self._PASSTHROUGH_MAX)
        inputs = proc(
            text=[new_text], images=self._resized_images,
            return_tensors="pt", padding=True,
        )
        self._update_visual_ranges(inputs["input_ids"][0])
        return inputs, new_text, new_images

    # ----- helpers --------------------------------------------------------

    def _update_visual_ranges(self, input_ids_1d: torch.Tensor):
        """Scan input_ids to find contiguous image-token blocks and update image_infos."""
        from gui_attention.attention import find_image_visual_ranges
        img_tok = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        ranges = find_image_visual_ranges(input_ids_1d, img_tok)
        for i, info in enumerate(self.image_infos):
            if i < len(ranges):
                info.visual_range = ranges[i]

    def get_image_grid_dims(self, image_grid_thw: torch.Tensor, merge_size: int):
        """Return list of (n_height, n_width) for each image after spatial merging."""
        dims = []
        for i in range(image_grid_thw.shape[0]):
            _, nh, nw = (image_grid_thw[i] // merge_size).tolist()
            dims.append((int(nh), int(nw)))
        return dims
