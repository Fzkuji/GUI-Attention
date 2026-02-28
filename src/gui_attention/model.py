"""Qwen2.5-VL + LoRA + ActionHead model wrapper.

Key difference from GUI-Actor: uses LLM last-layer hidden states (post-LLM,
fused text+visual context) rather than vision encoder embeddings (pre-LLM).
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from gui_attention.action_head import ActionHead
from gui_attention.constants import ADDITIONAL_SPECIAL_TOKENS


def build_model(
    model_name_or_path: str,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_target_modules: str = "q_proj,v_proj",
    torch_dtype=None,
    attn_implementation: str = "flash_attention_2",
    gradient_checkpointing: bool = True,
):
    """Build Qwen2.5-VL with LoRA and ActionHead.

    Returns:
        model: Qwen25VLWithActionHead (the wrapper).
        tokenizer: tokenizer with pointer tokens added.
        processor: image processor.
    """
    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    # Load backbone
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )
    backbone.config.use_cache = False

    if gradient_checkpointing and hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()

    # Add pointer special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    num_new = tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    if num_new > 0:
        backbone.resize_token_embeddings(len(tokenizer))
        ie = backbone.get_input_embeddings().weight.data
        oe = backbone.get_output_embeddings().weight.data
        ie[-num_new:] = ie[:-num_new].mean(0, keepdim=True)
        oe[-num_new:] = oe[:-num_new].mean(0, keepdim=True)

    # Store pointer token IDs on config for easy access
    backbone.config.pointer_start_token_id = tokenizer.convert_tokens_to_ids(
        ADDITIONAL_SPECIAL_TOKENS[0]
    )
    backbone.config.pointer_end_token_id = tokenizer.convert_tokens_to_ids(
        ADDITIONAL_SPECIAL_TOKENS[1]
    )
    backbone.config.pointer_pad_token_id = tokenizer.convert_tokens_to_ids(
        ADDITIONAL_SPECIAL_TOKENS[2]
    )

    # Apply LoRA
    target_modules = [m.strip() for m in lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    # Build action head (match backbone dtype)
    d_model = backbone.config.hidden_size
    action_head = ActionHead(d_model=d_model, projection_dim=d_model)
    action_head = action_head.to(torch_dtype)

    # Processor
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    processor.tokenizer = tokenizer

    model = Qwen25VLWithActionHead(backbone, action_head)
    return model, tokenizer, processor


class Qwen25VLWithActionHead(nn.Module):
    """Wrapper: Qwen2.5-VL (LoRA) + ActionHead.

    Forward:
        1. Run backbone with output_hidden_states=True.
        2. Extract visual hidden states from last layer at image_token positions.
        3. Extract anchor hidden states from last layer at pointer_pad positions.
        4. Run action head: MLP_V(visual) x MLP_T(anchor) → attention weights.
    """

    def __init__(self, backbone, action_head: ActionHead):
        super().__init__()
        self.backbone = backbone
        self.action_head = action_head

    @property
    def config(self):
        return self.backbone.config

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def forward(self, input_ids, attention_mask=None, pixel_values=None,
                image_grid_thw=None, **kwargs):
        """Run backbone forward pass with hidden state output."""
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            **kwargs,
        )
        return outputs

    def get_visual_hidden_states(self, outputs, input_ids, image_token_id):
        """Extract visual token hidden states from LLM last layer.

        Returns:
            (n_vis, d_model) tensor, or None if no visual tokens found.
        """
        hs = outputs.hidden_states[-1]  # (B, seq_len, d_model)
        mask = (input_ids[0] == image_token_id)
        if mask.sum() == 0:
            return None
        return hs[0][mask]  # (n_vis, d_model)

    def get_anchor_hidden_states(self, outputs, input_ids, pointer_pad_id):
        """Extract pointer_pad token hidden states from LLM last layer.

        Returns:
            (n_anchor, d_model) tensor, or None if not found.
        """
        hs = outputs.hidden_states[-1]
        if isinstance(pointer_pad_id, list):
            mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
            for pid in pointer_pad_id:
                mask |= (input_ids[0] == pid)
        else:
            mask = (input_ids[0] == pointer_pad_id)
        if mask.sum() == 0:
            return None
        return hs[0][mask]  # (n_anchor, d_model)

    def save_pretrained(self, path):
        """Save LoRA adapter + action head."""
        import os
        os.makedirs(path, exist_ok=True)
        # Save LoRA adapter weights
        self.backbone.save_pretrained(path)
        # Save action head separately
        torch.save(self.action_head.state_dict(), os.path.join(path, "action_head.pt"))

    @classmethod
    def load_pretrained(cls, path, base_model_name_or_path, torch_dtype=None,
                        attn_implementation="flash_attention_2", device="cuda:0"):
        """Load saved model (base + LoRA adapter + action head).

        Args:
            path: checkpoint directory with adapter_config.json and action_head.pt.
            base_model_name_or_path: original base model (e.g. Qwen/Qwen2.5-VL-3B-Instruct).
        """
        import os

        from peft import PeftModel

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        # Load base backbone
        backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        backbone.config.use_cache = False

        # Add pointer tokens (tokenizer from checkpoint already has them,
        # but we still need to resize embeddings to match)
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
        if len(tokenizer) != backbone.config.vocab_size:
            backbone.resize_token_embeddings(len(tokenizer))

        backbone.config.pointer_start_token_id = tokenizer.convert_tokens_to_ids(
            ADDITIONAL_SPECIAL_TOKENS[0]
        )
        backbone.config.pointer_end_token_id = tokenizer.convert_tokens_to_ids(
            ADDITIONAL_SPECIAL_TOKENS[1]
        )
        backbone.config.pointer_pad_token_id = tokenizer.convert_tokens_to_ids(
            ADDITIONAL_SPECIAL_TOKENS[2]
        )

        # Load LoRA adapter
        backbone = PeftModel.from_pretrained(backbone, path)

        # Load action head
        d_model = backbone.config.hidden_size
        action_head = ActionHead(d_model=d_model, projection_dim=d_model)
        head_path = os.path.join(path, "action_head.pt")
        if os.path.exists(head_path):
            action_head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
        action_head = action_head.to(device).to(torch_dtype)

        model = cls(backbone, action_head)
        return model, tokenizer
