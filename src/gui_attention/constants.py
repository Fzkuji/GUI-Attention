"""Shared constants for saccade foveation."""

IGNORE_INDEX = -100  # HuggingFace default for ignored labels in CrossEntropyLoss

# ---------------------------------------------------------------------------
# Pointer special tokens (from GUI-Actor, no gui_aima dependency)
# ---------------------------------------------------------------------------
DEFAULT_POINTER_START_TOKEN = "<|pointer_start|>"
DEFAULT_POINTER_END_TOKEN = "<|pointer_end|>"
DEFAULT_POINTER_PAD_TOKEN = "<|pointer_pad|>"

# Dual-action tokens: look (explore) vs click (commit)
DEFAULT_LOOK_PAD_TOKEN = "<|look_pad|>"
DEFAULT_CLICK_PAD_TOKEN = "<|click_pad|>"

ADDITIONAL_SPECIAL_TOKENS = [
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
    DEFAULT_LOOK_PAD_TOKEN,
    DEFAULT_CLICK_PAD_TOKEN,
]

# ---------------------------------------------------------------------------
# Chat template and system message (from GUI-Actor)
# ---------------------------------------------------------------------------
GROUNDING_SYSTEM_MESSAGE = (
    "You are a helpful assistant. You will receive a GUI screenshot along with a text instruction. "
    "Your task is to identify the UI element that matches the instruction and click on it."
)

CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "{% endif %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}<|im_end|>\n"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}"
    "Picture {{ image_count.value }}: "
    "{% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}"
    "Video {{ video_count.value }}: "
    "{% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}"
    "{{ content['text'] }}"
    "{% endif %}"
    "{% endfor %}"
    "<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)

# ---------------------------------------------------------------------------
# Resolution levels (simplified: low + high)
# ---------------------------------------------------------------------------
LOW_RES_MAX_PIXELS = 400_000     # ~480 tokens, 15×8 grid for 1080p
HIGH_RES_MAX_PIXELS = 5_720_064  # Same as GUI-Actor (~5.7M)

# ---------------------------------------------------------------------------
# Placeholder suffix appended after each round's image+instruction
# ---------------------------------------------------------------------------
# Legacy (single pointer_pad for backward compat)
PLACEHOLDER_SUFFIX = (
    "<|im_start|>assistant<|recipient|>os\n"
    "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
)

# Dual-action suffixes: look (explore) vs click (commit)
LOOK_SUFFIX = (
    "<|im_start|>assistant\n"
    "The target is not in the current view. Let me look at another region."
    "<|pointer_start|><|look_pad|><|pointer_end|>"
    "<|im_end|>\n"
)

CLICK_SUFFIX = (
    "<|im_start|>assistant\n"
    "I found the target in this region."
    "<|pointer_start|><|click_pad|><|pointer_end|>"
    "<|im_end|>\n"
)

# Round 0: always look (explore the full image first)
ROUND0_SUFFIX = (
    "<|im_start|>assistant\n"
    "Let me examine the screen to find the target."
    "<|pointer_start|><|look_pad|><|pointer_end|>"
    "<|im_end|>\n"
)
