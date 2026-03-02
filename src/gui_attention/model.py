"""Qwen2.5-VL + LoRA + ActionHead model wrapper.

Following GUI-Actor's design: visual encoder embeddings (pre-LLM) for the
action head's visual input, LLM last-layer hidden states for anchor/pointer.
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
    use_lora: bool = True,
):
    """Build Qwen2.5-VL with optional LoRA and ActionHead.

    Args:
        use_lora: If True, apply LoRA (default). If False, full parameter fine-tuning.

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

    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable()
        if hasattr(backbone, "enable_input_require_grads"):
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

    if use_lora:
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
    else:
        # Full parameter fine-tuning: all parameters trainable
        for param in backbone.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in backbone.parameters())
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"Full fine-tuning: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")

    # Build action head (match backbone dtype)
    d_model = getattr(backbone.config, "hidden_size", None) or backbone.config.text_config.hidden_size
    action_head = ActionHead(d_model=d_model, projection_dim=d_model)
    action_head = action_head.to(torch_dtype)

    # Processor
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    processor.tokenizer = tokenizer

    model = Qwen25VLWithActionHead(backbone, action_head)
    model._use_lora = use_lora
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

    def _get_unwrapped_backbone(self):
        """Unwrap PEFT layers to access base Qwen2.5-VL model."""
        backbone = self.backbone
        if hasattr(backbone, 'base_model'):
            base = backbone.base_model
            if hasattr(base, 'model'):
                base = base.model
        else:
            base = backbone
        return base

    def compute_visual_embeds(self, input_ids, pixel_values, image_grid_thw):
        """Compute visual embeddings (pre-LLM) at image token positions.

        Returns inputs_embeds (B, seq_len, d_model) with visual features
        scattered at image_token positions. Extract visual-only via:
            visual_mask = (input_ids == image_token_id)
            visual_embeds = inputs_embeds[0][visual_mask[0]]
        """
        base = self._get_unwrapped_backbone()
        inputs_embeds = base.model.embed_tokens(input_ids)
        if pixel_values is not None:
            image_embeds = base.visual(
                pixel_values.to(base.visual.dtype),
                grid_thw=image_grid_thw,
            )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

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

    def extract_visual_embeds(self, input_ids, pixel_values, image_grid_thw):
        """Extract visual embeddings at image token positions (no grad).

        Returns (n_vis, d_model) tensor of visual features from the vision
        encoder, preserving spatial information better than last-layer hidden
        states. Runs visual encoder under no_grad (no extra memory cost).
        """
        base = self._get_unwrapped_backbone()
        with torch.no_grad():
            image_embeds = base.visual(
                pixel_values.to(base.visual.dtype),
                grid_thw=image_grid_thw,
            )
        # image_embeds is (n_total_vis_tokens, d_model), already flattened
        return image_embeds.to(self.action_head.mlp_v[0].weight.dtype)

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
        """Save model weights + action head."""
        import os
        os.makedirs(path, exist_ok=True)
        # Save backbone (LoRA adapter or full model)
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
        _vocab_size = getattr(backbone.config, 'vocab_size', None) or getattr(backbone.config.text_config, 'vocab_size', None)
        if _vocab_size is not None and len(tokenizer) != _vocab_size:
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

        # Load weights: detect LoRA (adapter_config.json) vs full model
        adapter_cfg = os.path.join(path, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            # LoRA checkpoint
            backbone = PeftModel.from_pretrained(backbone, path)
        else:
            # Full model checkpoint — load state dict directly
            model_file = os.path.join(path, "model.safetensors")
            if os.path.exists(model_file):
                from safetensors.torch import load_file
                state = load_file(model_file)
                backbone.load_state_dict(state, strict=False)
            else:
                # Try pytorch_model.bin
                bin_file = os.path.join(path, "pytorch_model.bin")
                if os.path.exists(bin_file):
                    state = torch.load(bin_file, map_location=device, weights_only=True)
                    backbone.load_state_dict(state, strict=False)

        # Load action head
        d_model = getattr(backbone.config, "hidden_size", None) or backbone.config.text_config.hidden_size
        action_head = ActionHead(d_model=d_model, projection_dim=d_model)
        head_path = os.path.join(path, "action_head.pt")
        if os.path.exists(head_path):
            action_head.load_state_dict(
                torch.load(head_path, map_location=device, weights_only=True),
                strict=False,  # allow loading old checkpoints without self-attention
            )
        action_head = action_head.to(device).to(torch_dtype)

        model = cls(backbone, action_head)
        return model, tokenizer
