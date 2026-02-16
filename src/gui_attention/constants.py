"""Shared constants for saccade foveation (v4, self-contained)."""

# ---------------------------------------------------------------------------
# Pointer special tokens (from GUI-Actor, no gui_aima dependency)
# ---------------------------------------------------------------------------
DEFAULT_POINTER_START_TOKEN = "<|pointer_start|>"
DEFAULT_POINTER_END_TOKEN = "<|pointer_end|>"
DEFAULT_POINTER_PAD_TOKEN = "<|pointer_pad|>"

ADDITIONAL_SPECIAL_TOKENS = [
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
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
LOW_RES_MAX_PIXELS = 1_003_520   # Qwen2.5-VL default (~1M)
HIGH_RES_MAX_PIXELS = 5_720_064  # Same as GUI-Actor (~5.7M)

# ---------------------------------------------------------------------------
# Placeholder suffix appended after each round's image+instruction
# ---------------------------------------------------------------------------
PLACEHOLDER_SUFFIX = (
    "<|im_start|>assistant<|recipient|>os\n"
    "pyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
)
