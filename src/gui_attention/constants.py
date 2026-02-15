"""Shared constants for multi-precision foveated attention."""

# ---------------------------------------------------------------------------
# Multi-precision levels (from coarse to fine)
# ---------------------------------------------------------------------------
PRECISION_LEVELS = [250_000, 1_003_520, 4_000_000, 14_000_000]
LEVEL_NAMES = ["low", "original", "high", "ultra_high"]
STOP_LEVELS = {2, 3}  # attending to these levels terminates the foveation loop


def precision_for_level(level: int) -> int:
    """Return max_pixels for a given precision level (0-based)."""
    return PRECISION_LEVELS[min(level, len(PRECISION_LEVELS) - 1)]


# ---------------------------------------------------------------------------
# Pointer tokens (originally from gui_aima.constants)
# ---------------------------------------------------------------------------
DEFAULT_POINTER_START_TOKEN = "<|pointer_start|>"
DEFAULT_POINTER_END_TOKEN = "<|pointer_end|>"
DEFAULT_POINTER_PAD_TOKEN = "<|pointer_pad|>"

ADDITIONAL_SPECIAL_TOKENS = [
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
]

GROUNDING_SYSTEM_MESSAGE = (
    "You are a GUI agent. Given a screenshot of the current GUI and a human "
    "instruction, your task is to locate the screen element that corresponds "
    "to the instruction. You should output a PyAutoGUI action that performs a "
    "click on the correct position. To indicate the click location, we will "
    "use some special tokens, which is used to refer to a visual patch later. "
    "For example, you can output: pyautogui.click(<your_special_token_here>)."
)

CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}<|im_end|>\n"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}"
    "{% endif %}"
    "{% endfor %}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


# ---------------------------------------------------------------------------
# Placeholder suffix appended after each round's image+instruction
# ---------------------------------------------------------------------------
PLACEHOLDER_SUFFIX = (
    "<|im_start|>assistant<|recipient|>os\n"
    "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
)
