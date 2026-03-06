"""Qwen2.5-VL + LoRA + DualActionHead model wrapper.

Dual head design:
  - LookHead: coarse exploration, decides where to crop
  - ClickHead: precise clicking, only on high-res crop

Visual encoder embeddings (pre-LLM) for the action heads' visual input,
LLM last-layer hidden states for anchor/pointer.
"""

import os

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from gui_attention.constants import ADDITIONAL_SPECIAL_TOKENS
from gui_attention.dual_head import DualActionHead


def _load_click_head_from_pointer(dual_head, checkpoint_path):
    """Load GUI-Actor pointer_head weights into ClickHead.

    GUI-Actor (microsoft/GUI-Actor-3B-Qwen2.5-VL) stores the pointer head as
    'multi_patch_pointer_head.*' with different naming than our _AttentionHead.

    Key mapping (GUI-Actor VisionHead_MultiPatch → our _AttentionHead):
        multi_patch_pointer_head.self_attention.* → click_head.self_attention.*
        multi_patch_pointer_head.layer_norm.*     → click_head.layer_norm.*
        multi_patch_pointer_head.projection_enc.* → click_head.mlp_v.*
        multi_patch_pointer_head.projection_dec.* → click_head.mlp_t.*
    """
    from safetensors.torch import load_file
    import glob

    # Find safetensors files
    if os.path.isdir(checkpoint_path):
        sf_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    else:
        sf_files = [checkpoint_path]

    pointer_state = {}
    for sf in sf_files:
        state = load_file(sf)
        for k, v in state.items():
            if k.startswith("multi_patch_pointer_head."):
                # Strip prefix
                suffix = k[len("multi_patch_pointer_head."):]
                # Rename: projection_enc → mlp_v, projection_dec → mlp_t
                if suffix.startswith("projection_enc."):
                    suffix = suffix.replace("projection_enc.", "mlp_v.", 1)
                elif suffix.startswith("projection_dec."):
                    suffix = suffix.replace("projection_dec.", "mlp_t.", 1)
                new_key = f"click_head.{suffix}"
                pointer_state[new_key] = v

    if pointer_state:
        missing, unexpected = dual_head.load_state_dict(pointer_state, strict=False)
        # Only look_head keys should be missing (expected)
        actual_missing = [k for k in missing if not k.startswith("look_head.")]
        print(f"  ClickHead loaded from GUI-Actor pointer_head: {len(pointer_state)} params")
        if actual_missing:
            print(f"  WARNING: unexpected missing click_head keys: {actual_missing}")
        if unexpected:
            print(f"  WARNING: unexpected keys: {unexpected}")
    else:
        print(f"  WARNING: No multi_patch_pointer_head.* keys found in {checkpoint_path}")


def build_model(
    model_name_or_path: str,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_target_modules: str = "q_proj,v_proj",
    torch_dtype=None,
    attn_implementation: str = "flash_attention_2",
    gradient_checkpointing: bool = True,
    use_lora: bool = True,
    click_head_from: str = None,
):
    """Build Qwen2.5-VL with optional LoRA and DualActionHead.

    Args:
        click_head_from: path to a GUI-Actor style checkpoint (safetensors)
            to load pointer head weights into ClickHead. The pointer head
            keys are mapped: pointer_head.* → click_head.*

    Returns:
        model: Qwen25VLWithDualHead (the wrapper).
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
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
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
        for param in backbone.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in backbone.parameters())
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"Full fine-tuning: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)")

    # Build dual action head (match backbone dtype)
    d_model = getattr(backbone.config, "hidden_size", None) or backbone.config.text_config.hidden_size
    dual_head = DualActionHead(d_model=d_model, projection_dim=d_model)

    # Load ClickHead from GUI-Actor pointer head if specified
    if click_head_from:
        _load_click_head_from_pointer(dual_head, click_head_from)

    dual_head = dual_head.to(torch_dtype)

    # Processor
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    processor.tokenizer = tokenizer

    model = Qwen25VLWithDualHead(backbone, dual_head)
    model._use_lora = use_lora
    return model, tokenizer, processor


class Qwen25VLWithDualHead(nn.Module):
    """Wrapper: Qwen2.5-VL (LoRA) + DualActionHead (LookHead + ClickHead).

    Forward:
        1. Run backbone with output_hidden_states=True.
        2. Extract visual hidden states from ViT (pre-LLM).
        3. Extract anchor hidden states from LLM last layer at pointer_pad positions.
        4. Run LookHead or ClickHead as needed.
    """

    def __init__(self, backbone, dual_head: DualActionHead):
        super().__init__()
        self.backbone = backbone
        self.dual_head = dual_head

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
        """Extract visual embeddings from ViT (pre-LLM, no grad).

        Returns (n_vis, d_model) tensor of visual features.
        """
        base = self._get_unwrapped_backbone()
        with torch.no_grad():
            image_embeds = base.visual(
                pixel_values.to(base.visual.dtype),
                grid_thw=image_grid_thw,
            )
        return image_embeds.to(self.dual_head.look_head.mlp_v[0].weight.dtype)

    def get_visual_hidden_states(self, outputs, input_ids, image_token_id):
        """Extract visual token hidden states from LLM last layer."""
        hs = outputs.hidden_states[-1]
        mask = (input_ids[0] == image_token_id)
        if mask.sum() == 0:
            return None
        return hs[0][mask]

    def get_anchor_hidden_states(self, outputs, input_ids, pointer_pad_id):
        """Extract pointer_pad token hidden states from LLM last layer."""
        hs = outputs.hidden_states[-1]
        if isinstance(pointer_pad_id, list):
            mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
            for pid in pointer_pad_id:
                mask |= (input_ids[0] == pid)
        else:
            mask = (input_ids[0] == pointer_pad_id)
        if mask.sum() == 0:
            return None
        return hs[0][mask]

    def save_pretrained(self, path):
        """Save model weights + dual head."""
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path)
        torch.save(self.dual_head.state_dict(), os.path.join(path, "dual_head.pt"))

    @classmethod
    def load_pretrained(cls, path, base_model_name_or_path, torch_dtype=None,
                        attn_implementation="flash_attention_2", device="cuda:0"):
        """Load saved model (base + LoRA adapter + dual head).

        Also supports loading old ActionHead checkpoints (backward compat).
        """
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

        # Add pointer tokens
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

        # Load weights: detect LoRA vs full model
        adapter_cfg = os.path.join(path, "adapter_config.json")
        if os.path.exists(adapter_cfg):
            backbone = PeftModel.from_pretrained(backbone, path)
        else:
            model_file = os.path.join(path, "model.safetensors")
            if os.path.exists(model_file):
                from safetensors.torch import load_file
                state = load_file(model_file)
                backbone.load_state_dict(state, strict=False)
            else:
                bin_file = os.path.join(path, "pytorch_model.bin")
                if os.path.exists(bin_file):
                    state = torch.load(bin_file, map_location=device, weights_only=True)
                    backbone.load_state_dict(state, strict=False)

        # Load dual head
        d_model = getattr(backbone.config, "hidden_size", None) or backbone.config.text_config.hidden_size
        dual_head = DualActionHead(d_model=d_model, projection_dim=d_model)

        dual_head_path = os.path.join(path, "dual_head.pt")
        old_head_path = os.path.join(path, "action_head.pt")

        if os.path.exists(dual_head_path):
            dual_head.load_state_dict(
                torch.load(dual_head_path, map_location=device, weights_only=True),
                strict=False,
            )
        elif os.path.exists(old_head_path):
            # Backward compat: load old ActionHead weights into LookHead
            old_state = torch.load(old_head_path, map_location=device, weights_only=True)
            look_state = {}
            for k, v in old_state.items():
                # Skip bbox_head and beta (old ActionHead specific)
                if k.startswith("bbox_head") or k == "beta":
                    continue
                look_state[f"look_head.{k}"] = v
            dual_head.load_state_dict(look_state, strict=False)
            print(f"  Loaded old ActionHead weights into LookHead (backward compat)")

        dual_head = dual_head.to(device).to(torch_dtype)

        model = cls(backbone, dual_head)
        return model, tokenizer
