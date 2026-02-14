"""MultiRoundInputBuilder: incrementally builds multi-round conversation tokens.

Each round's image is pre-processed at its own resolution so the processor
never re-sizes an image intended for a different precision level.
"""

from typing import Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info, smart_resize

from gui_aima.constants import chat_template, grounding_system_message

from gui_attention.constants import PLACEHOLDER_SUFFIX, precision_for_round


def _presize_image(image: Image.Image, max_pixels: int, min_pixels: int = 3136) -> Image.Image:
    """Pre-resize an image to the exact dimensions the Qwen2.5-VL processor
    would use, so a subsequent processor call with large max_pixels won't
    resize it further."""
    w, h = image.size
    new_h, new_w = smart_resize(h, w, max_pixels=max_pixels, min_pixels=min_pixels)
    if (new_w, new_h) != (w, h):
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image


class MultiRoundInputBuilder:
    """Incrementally builds the multi-round conversation and tokenises it.

    Round 0 (full image):
        system + user[image, instruction] + placeholder

    Round k (crop):
        ... previous ... + crop_image [Zoomed ...] + placeholder
    """

    def __init__(self, model_path: str, tokenizer, min_pixels: int):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.min_pixels = min_pixels
        self._processor_cache: Dict[int, AutoProcessor] = {}
        # Pre-resized images (one per round) – kept so they can be passed
        # together to a single processor call without further resizing.
        self._resized_images: List[Image.Image] = []

    def _get_processor(self, max_pixels: int):
        if max_pixels not in self._processor_cache:
            p = AutoProcessor.from_pretrained(
                self.model_path, min_pixels=self.min_pixels, max_pixels=max_pixels,
            )
            p.tokenizer = self.tokenizer
            self._processor_cache[max_pixels] = p
        return self._processor_cache[max_pixels]

    # Large enough that the processor won't resize our pre-resized images.
    _PASSTHROUGH_MAX = 20_000_000

    def build_round0(self, image_or_path, instruction: str, max_pixels: int):
        """Build round-0 inputs (full image)."""
        if isinstance(image_or_path, dict):
            image_or_path = image_or_path["image_path"]
        conv = [
            {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image_or_path},
                {"type": "text", "text": instruction},
            ]},
        ]
        text = self._get_processor(max_pixels).apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False, chat_template=chat_template,
        )
        text += PLACEHOLDER_SUFFIX
        images, _ = process_vision_info(conv)

        # Pre-resize the round-0 image so its token count is locked in.
        resized_r0 = _presize_image(images[0], max_pixels, self.min_pixels)
        self._resized_images = [resized_r0]

        inputs = self._get_processor(max_pixels)(
            text=[text], images=[resized_r0], return_tensors="pt", padding=True,
        )
        return inputs, text, images  # return original images list for extend

    def extend_with_crop(self, prev_text: str, prev_images: list,
                         crop_pil: Image.Image, crop_bbox: tuple,
                         round_idx: int):
        """Append a crop round to the existing conversation.

        The crop is pre-resized at this round's resolution, then ALL images
        (each already at its own target resolution) are passed to a single
        processor with a very large max_pixels so no further resizing occurs.
        """
        max_px = precision_for_round(round_idx)
        # Use <|vision_start|><|image_pad|><|vision_end|> instead of raw <image>
        # so the processor correctly expands the placeholder to match features.
        zoom_text = (
            f"\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"[Zoomed region round {round_idx + 1} around "
            f"({crop_bbox[0]:.2f},{crop_bbox[1]:.2f})-({crop_bbox[2]:.2f},{crop_bbox[3]:.2f})]"
            f"<|im_end|>\n"
            + PLACEHOLDER_SUFFIX
        )
        new_text = prev_text + zoom_text
        new_images = prev_images + [crop_pil]

        # Pre-resize the crop at this round's resolution
        resized_crop = _presize_image(crop_pil, max_px, self.min_pixels)
        self._resized_images.append(resized_crop)

        # Process with a large max_pixels so the processor doesn't resize
        # any of our pre-resized images.
        proc = self._get_processor(self._PASSTHROUGH_MAX)
        inputs = proc(
            text=[new_text], images=self._resized_images,
            return_tensors="pt", padding=True,
        )
        return inputs, new_text, new_images
