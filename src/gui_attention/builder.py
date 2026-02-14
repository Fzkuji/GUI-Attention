"""MultiRoundInputBuilder: incrementally builds multi-round conversation tokens."""

from typing import Dict

from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from gui_aima.constants import chat_template, grounding_system_message

from gui_attention.constants import PLACEHOLDER_SUFFIX, precision_for_round


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

    def _get_processor(self, max_pixels: int):
        if max_pixels not in self._processor_cache:
            p = AutoProcessor.from_pretrained(
                self.model_path, min_pixels=self.min_pixels, max_pixels=max_pixels,
            )
            p.tokenizer = self.tokenizer
            self._processor_cache[max_pixels] = p
        return self._processor_cache[max_pixels]

    def build_round0(self, image_or_path, instruction: str, max_pixels: int):
        """Build round-0 inputs (full image).

        image_or_path: either an image file path (str) or a dict with 'image_path' key.
        """
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
        inputs = self._get_processor(max_pixels)(
            text=[text], images=images, return_tensors="pt", padding=True,
        )
        return inputs, text, images

    def extend_with_crop(self, prev_text: str, prev_images: list,
                         crop_pil: Image.Image, crop_bbox: tuple,
                         round_idx: int):
        """Append a crop round to the existing conversation."""
        max_px = precision_for_round(round_idx)
        zoom_text = (
            f"\n<|im_start|>user\n<image>"
            f"[Zoomed region round {round_idx + 1} around "
            f"({crop_bbox[0]:.2f},{crop_bbox[1]:.2f})-({crop_bbox[2]:.2f},{crop_bbox[3]:.2f})]"
            f"<|im_end|>\n"
            + PLACEHOLDER_SUFFIX
        )
        new_text = prev_text + zoom_text
        new_images = prev_images + [crop_pil]
        proc = self._get_processor(max_px)
        inputs = proc(text=[new_text], images=new_images, return_tensors="pt", padding=True)
        return inputs, new_text, new_images
