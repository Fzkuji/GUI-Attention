"""Shared helpers for free-form reasoning with structured action spans."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from transformers import StoppingCriteria

from gui_attention.constants import (
    DEFAULT_CLICK_PAD_TOKEN,
    DEFAULT_LOOK_PAD_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_START_TOKEN,
)

THOUGHT_PREFIX = "Thought: "
ACTION_PREFIX = "Action: "

LOOK_ACTION_SPAN = (
    f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_LOOK_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
)
CLICK_ACTION_SPAN = (
    f"{DEFAULT_POINTER_START_TOKEN}{DEFAULT_CLICK_PAD_TOKEN}{DEFAULT_POINTER_END_TOKEN}"
)

REASONING_ASSISTANT_PREFIX = "<|im_start|>assistant\nThought: "

ROUND0_REASONING_GUIDE = (
    "\nRespond as the assistant using exactly two lines:\n"
    f"{THOUGHT_PREFIX}<brief reasoning about which region to inspect next>\n"
    f"{ACTION_PREFIX}{LOOK_ACTION_SPAN}\n"
    "This is the initial low-resolution overview, so you must inspect a high-resolution region next."
)

CROP_REASONING_GUIDE = (
    "\nRespond as the assistant using exactly two lines:\n"
    f"{THOUGHT_PREFIX}<brief reasoning>\n"
    f"{ACTION_PREFIX}{LOOK_ACTION_SPAN} or {CLICK_ACTION_SPAN}\n"
    "Use the look action when you need another close-up. Use the click action only when the target is clear enough to click now."
)


@dataclass
class ParsedReasoningAction:
    raw_content: str
    used_content: str
    thought: str
    action: str
    format_ok: bool
    parse_failed: bool


def get_reasoning_guide(*, allow_click: bool) -> str:
    return CROP_REASONING_GUIDE if allow_click else ROUND0_REASONING_GUIDE


def decode_reasoning_content(tokenizer, generated_token_ids: Sequence[int]) -> str:
    decoded = tokenizer.decode(
        list(generated_token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    decoded = decoded.replace("<|im_end|>", "").strip()
    if not decoded.startswith(THOUGHT_PREFIX):
        decoded = f"{THOUGHT_PREFIX}{decoded}"
    return decoded.strip()


def normalize_assistant_content(content: str) -> str:
    text = content.strip()
    if text.startswith("<|im_start|>assistant"):
        text = text[len("<|im_start|>assistant"):].lstrip("\n")
    text = text.replace("<|im_end|>", "").strip()
    return text


def wrap_assistant_content(content: str) -> str:
    text = normalize_assistant_content(content)
    return f"<|im_start|>assistant\n{text}<|im_end|>\n"


def canonical_assistant_content(thought: str, action: str) -> str:
    thought = " ".join((thought or "").split()).strip()
    if not thought:
        thought = (
            "I should inspect another region more carefully."
            if action == "look"
            else "I have enough evidence to click the target now."
        )
    action_span = LOOK_ACTION_SPAN if action == "look" else CLICK_ACTION_SPAN
    return f"{THOUGHT_PREFIX}{thought}\n{ACTION_PREFIX}{action_span}"


def parse_reasoning_action(content: str, *, allow_click: bool) -> ParsedReasoningAction:
    text = normalize_assistant_content(content)

    look_hits = text.count(LOOK_ACTION_SPAN)
    click_hits = text.count(CLICK_ACTION_SPAN)
    total_hits = look_hits + click_hits

    action_line_idx = text.rfind(f"\n{ACTION_PREFIX}")
    if action_line_idx < 0:
        action_line_idx = text.find(ACTION_PREFIX)
    thought_block = text[:action_line_idx] if action_line_idx >= 0 else text
    if THOUGHT_PREFIX in thought_block:
        thought = thought_block.split(THOUGHT_PREFIX, 1)[1].strip()
    else:
        thought = thought_block.strip()

    format_ok = (
        text.startswith(THOUGHT_PREFIX)
        and action_line_idx >= 0
        and bool(thought)
        and total_hits == 1
    )

    parse_failed = total_hits != 1
    action = "look"
    if look_hits == 1 and click_hits == 0:
        action = "look"
    elif click_hits == 1 and look_hits == 0:
        action = "click"

    if action == "click" and not allow_click:
        parse_failed = True
        format_ok = False
        action = "look"

    if total_hits != 1:
        action = "look"

    used_content = canonical_assistant_content(thought, action)
    return ParsedReasoningAction(
        raw_content=text,
        used_content=used_content,
        thought=thought,
        action=action,
        format_ok=format_ok and not parse_failed,
        parse_failed=parse_failed or not bool(thought),
    )


class ActionSpanStoppingCriteria(StoppingCriteria):
    """Stop after the model emits a full action span."""

    def __init__(
        self,
        *,
        prompt_len: int,
        pointer_end_token_id: int,
        allowed_action_token_ids: Iterable[int],
    ):
        super().__init__()
        self.prompt_len = prompt_len
        self.pointer_end_token_id = pointer_end_token_id
        self.allowed_action_token_ids = {
            int(t) for t in allowed_action_token_ids if t is not None
        }

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        if input_ids.shape[0] != 1:
            return False
        generated = input_ids[0, self.prompt_len:].tolist()
        if not generated:
            return False
        if generated[-1] != self.pointer_end_token_id:
            return False
        return any(t in self.allowed_action_token_ids for t in generated)
