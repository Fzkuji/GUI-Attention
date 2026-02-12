"""
GUI-AIMA training with LoRA + pointer-token attention supervision.
Uses GUI-AIMA's exact training logic (dataset, model, loss) but with LoRA for memory efficiency.
"""
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import os
import json
import torch
import transformers
from PIL import ImageFile
from transformers import AutoProcessor
from peft import LoraConfig, get_peft_model

try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
    apply_liger_kernel_to_qwen2_vl()
except ImportError:
    pass

from gui_aima.dataset import LazySupervisedDataset
from gui_aima.trainer import AGUVISTrainer, rank0_print, safe_save_model_for_hf_trainer, EmptyCacheCallback
from gui_aima.utils import dump_args_to_json
from gui_aima.constants import (
    IGNORE_INDEX,
    ADDITIONAL_SPECIAL_TOKENS,
    DEFAULT_POINTER_START_TOKEN,
    DEFAULT_POINTER_END_TOKEN,
    DEFAULT_POINTER_PAD_TOKEN,
)
from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer

torch.multiprocessing.set_sharing_strategy("file_system")
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    model_type: str = field(default="qwen25vl")
    weighting: str = field(default=None)
    number_of_points: int = field(default=1)
    # LoRA
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    min_pixels: Optional[int] = field(default=3136)
    max_pixels: Optional[int] = field(default=1003520)  # 1024 * 28 * 28
    max_conv_turns: Optional[int] = field(default=10)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=8192)
    gradient_checkpointing: bool = field(default=True)
    pointer_loss_weight: float = field(default=1.0)
    lm_loss_weight: float = field(default=1.0)
    empty_cache_every_n_steps: int = field(default=15)


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = len(tokenizer)
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = new_vocab_size
    else:
        model.config.vocab_size = new_vocab_size
    model.vocab_size = new_vocab_size
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def update_pointer_token_ids(model_config, tokenizer, number_of_points):
    model_config.pointer_start_token_id = tokenizer.encode(DEFAULT_POINTER_START_TOKEN)[0]
    model_config.pointer_end_token_id = tokenizer.encode(DEFAULT_POINTER_END_TOKEN)[0]
    model_config.pointer_pad_token_id = tokenizer.encode(DEFAULT_POINTER_PAD_TOKEN)[0]
    rank0_print(f"Pointer token ids: pad={model_config.pointer_pad_token_id}, "
                f"start={model_config.pointer_start_token_id}, end={model_config.pointer_end_token_id}")


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[:self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[:self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = {
            "input_ids": input_ids,
            "labels": labels.long() if labels.dtype == torch.int32 else labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if "pixel_values" in instances[0]:
            batch["pixel_values"] = torch.concat([instance["pixel_values"] for instance in instances], dim=0)
            batch["image_grid_thw"] = torch.concat([instance["image_grid_thw"] for instance in instances], dim=0)
        if "coordinates" in instances[0]:
            batch["coordinates"] = [instance["coordinates"] for instance in instances]
            batch["visual_token_indices_of_coordinates"] = [instance["visual_token_indices_of_coordinates"] for instance in instances]
        if "multi_patch_labels" in instances[0]:
            batch["multi_patch_labels"] = [instance["multi_patch_labels"] for instance in instances]
        return batch


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load model
    rank0_print(f"Loading model from {model_args.model_name_or_path}...")
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="sdpa",
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False
    model.reset_loss_weights(
        pointer_loss_weight=training_args.pointer_loss_weight,
        lm_loss_weight=training_args.lm_loss_weight,
    )
    if model_args.weighting is not None:
        model.set_attention_args(weighting=model_args.weighting)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            model.get_input_embeddings().register_forward_hook(
                lambda module, input, output: output.requires_grad_(True)
            )

    # Tokenizer + special tokens
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS},
        tokenizer=tokenizer,
        model=model,
    )
    update_pointer_token_ids(model.config, tokenizer, model_args.number_of_points)

    # Apply LoRA
    rank0_print(f"Applying LoRA: r={model_args.lora_r}, alpha={model_args.lora_alpha}")
    
    # Freeze all first
    for p in model.parameters():
        p.requires_grad = False
    
    # LoRA on attention layers
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # Also unfreeze: pointer head (multi_patch_pointer_head_attention), new token embeddings
    for name, param in model.named_parameters():
        if "multi_patch_pointer_head_attention" in name:
            param.requires_grad = True
        if "embed_tokens" in name:
            param.requires_grad = True
        if "lm_head" in name:
            param.requires_grad = True
    
    model.print_trainable_parameters()
    
    # Mask embedding gradients for non-new tokens
    n_new_tokens = len(ADDITIONAL_SPECIAL_TOKENS)
    for name, param in model.named_parameters():
        if name.endswith("embed_tokens.weight"):
            def mask_grad(grad, n=n_new_tokens):
                grad[:-n] = 0.0
                return grad
            param.register_hook(mask_grad)
            break

    # Processor
    data_args.processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
    )
    data_args.processor.tokenizer = tokenizer

    # Dataset
    os.makedirs(training_args.output_dir, exist_ok=True)
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        processor=data_args.processor,
        data_path=data_args.data_path,
        data_args=data_args,
        number_of_points=model_args.number_of_points,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Trainer
    trainer = AGUVISTrainer(
        model=model,
        processing_class=data_args.processor,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[EmptyCacheCallback(every_n_steps=training_args.empty_cache_every_n_steps)],
    )

    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    
    # Save LoRA adapter + tokenizer
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
