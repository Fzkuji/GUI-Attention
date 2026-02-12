"""
Phase 2: Iterative Foveated Inference with GUI-AIMA trained model.

Key innovation: Instead of processing a single high-resolution image (~18,800 visual tokens),
we perform 3 rounds of progressive zoom-in, each with a low-token-count image:

  Round 1: Low-res full image (~200 tokens) → attention locates rough region
  Round 2: Medium-res crop (~300 tokens) → refines location
  Round 3: High-res crop (~200 tokens) → precise grounding

Total: ~700 tokens vs ~18,800 tokens (96% reduction)

Uses GUI-AIMA's pointer-token attention mechanism for grounding at each round.
"""

import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, LogitsProcessorList

# GUI-AIMA imports
import sys
sys.path.insert(0, "/root/GUI-AIMA/src")
from gui_aima.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from gui_aima.inference import (
    calculate_attention_from_qk,
    get_prediction_region_point,
    ForceFollowTokensLogitsProcessorSimple,
)
from gui_aima.constants import chat_template, grounding_system_message


@dataclass
class FoveationConfig:
    """Configuration for foveated inference."""
    # Round 1: Periphery (full image, very low res)
    periphery_max_pixels: int = 56 * 56 * 28 * 28  # ~200 tokens after merge
    
    # Round 2: Parafovea (cropped region, medium res)  
    parafovea_crop_ratio: float = 0.4  # crop 40% of the image around fixation
    parafovea_max_pixels: int = 56 * 56 * 28 * 28  # ~300 tokens
    
    # Round 3: Fovea (smaller crop, higher res)
    fovea_crop_ratio: float = 0.15  # crop 15% of the image
    fovea_max_pixels: int = 56 * 56 * 28 * 28  # ~200 tokens
    
    # Shared
    min_pixels: int = 4 * 28 * 28  # minimum
    num_rounds: int = 3
    use_placeholder: bool = True
    topk: int = 3  # number of candidate regions from attention


class FoveatedInference:
    """
    Iterative foveated inference using a GUI-AIMA trained model.
    
    The model is used as-is (no additional training needed for Phase 2).
    We just control the input resolution and crop region at each round.
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[FoveationConfig] = None,
        device: str = "cuda",
    ):
        self.config = config or FoveationConfig()
        self.device = device
        
        # Load GUI-AIMA model
        print(f"Loading GUI-AIMA model from {model_path}...")
        self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(device).eval()
        
        self.tokenizer = self.model.config._name_or_path  # will use processor's tokenizer
        self.base_processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.periphery_max_pixels,
        )
        
        # Logits processor for pointer tokens
        if isinstance(self.model.config.pointer_pad_token_id, list):
            self.n_points = len(self.model.config.pointer_pad_token_id)
        else:
            self.n_points = 1
        
        self.logits_processor = ForceFollowTokensLogitsProcessorSimple(
            tokenizer=self.base_processor.tokenizer,
            number_of_points=self.n_points,
        )
        
        print(f"Model loaded. Foveation config: {self.config.num_rounds} rounds")
    
    def _make_processor(self, max_pixels: int) -> AutoProcessor:
        """Create a processor with specific max_pixels."""
        proc = AutoProcessor.from_pretrained(
            self.model.config._name_or_path,
            min_pixels=self.config.min_pixels,
            max_pixels=max_pixels,
        )
        return proc
    
    def _crop_image(
        self, image: Image.Image, center_x: float, center_y: float, crop_ratio: float
    ) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
        """
        Crop a region around (center_x, center_y) with given ratio.
        Returns cropped image and bbox in original image coordinates [0,1].
        """
        W, H = image.size
        crop_w = int(W * crop_ratio)
        crop_h = int(H * crop_ratio)
        
        # Center the crop, clamp to image bounds
        cx_px = int(center_x * W)
        cy_px = int(center_y * H)
        
        x1 = max(0, cx_px - crop_w // 2)
        y1 = max(0, cy_px - crop_h // 2)
        x2 = min(W, x1 + crop_w)
        y2 = min(H, y1 + crop_h)
        
        # Adjust if hit boundary
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
        
        cropped = image.crop((x1, y1, x2, y2))
        
        # Bbox in normalized coordinates
        bbox = (x1 / W, y1 / H, x2 / W, y2 / H)
        return cropped, bbox
    
    def _single_round(
        self,
        image: Image.Image,
        instruction: str,
        max_pixels: int,
        crop_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    ) -> Dict:
        """
        Run one round of inference using GUI-AIMA's attention mechanism.
        
        Returns:
            dict with keys:
                - pred_x, pred_y: predicted point in ORIGINAL image coordinates [0,1]
                - attn_scores: raw attention scores
                - n_visual_tokens: number of visual tokens used
                - topk_points: top-k candidate points (in original coords)
        """
        # Build conversation
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": grounding_system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        
        # Prepare inputs with controlled resolution
        text = self.base_processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False,
            chat_template=chat_template,
        )
        
        if self.config.use_placeholder:
            text += "<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)"
        
        image_inputs, _ = process_vision_info(conversation)
        
        # Create processor with appropriate max_pixels for this round
        processor = AutoProcessor.from_pretrained(
            self.model.config._name_or_path,
            min_pixels=self.config.min_pixels,
            max_pixels=max_pixels,
        )
        processor.tokenizer = self.base_processor.tokenizer
        
        inputs = processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.device)
        
        input_ids = inputs["input_ids"][0]
        
        # Count visual tokens
        visual_mask = (input_ids == self.model.config.image_token_id)
        visual_indices = torch.nonzero(visual_mask, as_tuple=False).squeeze(-1)
        n_visual = visual_indices.shape[0]
        
        # Generate with attention
        with torch.no_grad():
            position_ids, _ = self.model.get_rope_index(
                input_ids=inputs["input_ids"],
                image_grid_thw=inputs["image_grid_thw"],
                video_grid_thw=None,
                attention_mask=inputs["attention_mask"],
            )
            
            results = self.model.generate(
                **inputs,
                max_new_tokens=1 if self.config.use_placeholder else 2048,
                logits_processor=LogitsProcessorList([self.logits_processor]),
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=True,
                temperature=0.1,
            )
        
        # Extract pointer attention
        pointer_pad_mask = torch.isin(
            input_ids,
            torch.tensor(self.model.config.pointer_pad_token_id, device=input_ids.device),
        )
        target_indices = torch.nonzero(pointer_pad_mask, as_tuple=False).squeeze(-1)
        
        if target_indices.numel() == 0 or visual_indices.numel() == 0:
            # Fallback: return center
            return {
                "pred_x": 0.5, "pred_y": 0.5,
                "n_visual_tokens": n_visual,
                "topk_points": [(0.5, 0.5)],
            }
        
        # Query indices: text tokens between last visual and pointer_start
        query_start = visual_indices[-1] + 1
        ps_mask = (input_ids == self.model.config.pointer_start_token_id)
        ps_positions = torch.nonzero(ps_mask, as_tuple=False).squeeze(-1)
        query_end = ps_positions[0].item() if ps_positions.numel() > 0 else len(input_ids)
        query_indices = torch.arange(query_start, query_end, device=input_ids.device)
        
        if getattr(self.model.config, 'part_query_weighting', False):
            query_indices = query_indices[:-12]
        
        merged_indices = torch.cat([query_indices, target_indices], dim=0)
        
        # QK-recompute attention
        calculated_attention = calculate_attention_from_qk(
            model=self.model,
            all_hidden_states=results.hidden_states,
            all_position_ids=position_ids,
            query_indices=merged_indices,
            all_attention_mask=inputs["attention_mask"],
        )
        
        # Query importance weighting (cosine similarity)
        all_layer_hs = torch.stack(results.hidden_states[0][1:], dim=0)
        sample_hs = all_layer_hs[:, 0, :, :]
        query_hs = F.normalize(sample_hs[:, query_indices, :], dim=-1)
        visual_hs = F.normalize(sample_hs[:, visual_indices, :], dim=-1)
        sim = torch.einsum('lqd,lvd->lqv', query_hs, visual_hs)
        
        topk_query_indices = None
        global_pattern = None
        
        if not getattr(self.model.config, 'kl_query_weighting', False):
            agg = sim.sum(dim=-1).sum(dim=0)
            k = getattr(self.model.config, 'query_topk', 5)
            _, topk_query_indices = torch.topk(agg, min(k, len(query_indices)), largest=True)
        else:
            global_pattern = sim.sum(dim=-1).sum(dim=0).softmax(dim=-1)
        
        # Aggregate attention through pointer head
        attn_scores, _ = self.model.multi_patch_pointer_head_attention(
            query_indices, visual_indices, target_indices,
            calculated_attention[0], topk_query_indices, global_pattern,
            batch_idx=0,
        )
        
        # Get spatial dimensions
        merge_size = self.model.visual.spatial_merge_size
        _, n_height, n_width = (inputs["image_grid_thw"][0] // merge_size).tolist()
        
        # Extract prediction point
        best_point, region_points, region_scores, _ = get_prediction_region_point(
            attn_scores, int(n_width), int(n_height), return_all_regions=True,
        )
        
        # Map from crop coordinates back to original image coordinates
        bx1, by1, bx2, by2 = crop_bbox
        pred_x = bx1 + best_point[0] * (bx2 - bx1)
        pred_y = by1 + best_point[1] * (by2 - by1)
        
        topk_points = []
        for px, py in region_points[:self.config.topk]:
            ox = bx1 + px * (bx2 - bx1)
            oy = by1 + py * (by2 - by1)
            topk_points.append((ox, oy))
        
        return {
            "pred_x": pred_x,
            "pred_y": pred_y,
            "n_visual_tokens": n_visual,
            "topk_points": topk_points,
            "topk_scores": region_scores[:self.config.topk] if region_scores else [],
            "attn_max": attn_scores[0].max().item() if attn_scores is not None else 0,
        }
    
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        instruction: str,
        verbose: bool = True,
    ) -> Tuple[float, float]:
        """
        Iterative foveated inference.
        
        Args:
            image: Full-resolution input image
            instruction: Grounding instruction (e.g., "Click the submit button")
            verbose: Print per-round statistics
            
        Returns:
            (pred_x, pred_y): Predicted click point in [0, 1] coordinates
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        total_tokens = 0
        results_per_round = []
        
        # === Round 1: Periphery (full image, low res) ===
        r1 = self._single_round(
            image, instruction,
            max_pixels=self.config.periphery_max_pixels,
            crop_bbox=(0.0, 0.0, 1.0, 1.0),
        )
        total_tokens += r1["n_visual_tokens"]
        results_per_round.append(r1)
        
        if verbose:
            print(f"  R1 (periphery): pred=({r1['pred_x']:.4f}, {r1['pred_y']:.4f}), "
                  f"tokens={r1['n_visual_tokens']}, attn_max={r1.get('attn_max', 0):.4f}")
        
        if self.config.num_rounds < 2:
            return r1["pred_x"], r1["pred_y"]
        
        # === Round 2: Parafovea (crop around R1 prediction) ===
        fx, fy = r1["pred_x"], r1["pred_y"]
        crop2, bbox2 = self._crop_image(image, fx, fy, self.config.parafovea_crop_ratio)
        
        r2 = self._single_round(
            crop2, instruction,
            max_pixels=self.config.parafovea_max_pixels,
            crop_bbox=bbox2,
        )
        total_tokens += r2["n_visual_tokens"]
        results_per_round.append(r2)
        
        if verbose:
            print(f"  R2 (parafovea): pred=({r2['pred_x']:.4f}, {r2['pred_y']:.4f}), "
                  f"tokens={r2['n_visual_tokens']}, crop={bbox2}")
        
        if self.config.num_rounds < 3:
            return r2["pred_x"], r2["pred_y"]
        
        # === Round 3: Fovea (smaller crop around R2 prediction) ===
        fx, fy = r2["pred_x"], r2["pred_y"]
        crop3, bbox3 = self._crop_image(image, fx, fy, self.config.fovea_crop_ratio)
        
        r3 = self._single_round(
            crop3, instruction,
            max_pixels=self.config.fovea_max_pixels,
            crop_bbox=bbox3,
        )
        total_tokens += r3["n_visual_tokens"]
        results_per_round.append(r3)
        
        if verbose:
            print(f"  R3 (fovea):     pred=({r3['pred_x']:.4f}, {r3['pred_y']:.4f}), "
                  f"tokens={r3['n_visual_tokens']}, crop={bbox3}")
            print(f"  Total visual tokens: {total_tokens}")
        
        return r3["pred_x"], r3["pred_y"]
    
    def predict_with_details(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict:
        """
        Same as predict() but returns full details for analysis.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        details = {
            "rounds": [],
            "total_tokens": 0,
            "final_pred": None,
        }
        
        # Round 1
        r1 = self._single_round(
            image, instruction,
            max_pixels=self.config.periphery_max_pixels,
            crop_bbox=(0.0, 0.0, 1.0, 1.0),
        )
        details["rounds"].append({"name": "periphery", **r1, "crop_bbox": (0, 0, 1, 1)})
        details["total_tokens"] += r1["n_visual_tokens"]
        
        if self.config.num_rounds >= 2:
            crop2, bbox2 = self._crop_image(image, r1["pred_x"], r1["pred_y"], self.config.parafovea_crop_ratio)
            r2 = self._single_round(crop2, instruction, self.config.parafovea_max_pixels, bbox2)
            details["rounds"].append({"name": "parafovea", **r2, "crop_bbox": bbox2})
            details["total_tokens"] += r2["n_visual_tokens"]
        
        if self.config.num_rounds >= 3:
            crop3, bbox3 = self._crop_image(image, r2["pred_x"], r2["pred_y"], self.config.fovea_crop_ratio)
            r3 = self._single_round(crop3, instruction, self.config.fovea_max_pixels, bbox3)
            details["rounds"].append({"name": "fovea", **r3, "crop_bbox": bbox3})
            details["total_tokens"] += r3["n_visual_tokens"]
        
        last = details["rounds"][-1]
        details["final_pred"] = (last["pred_x"], last["pred_y"])
        
        return details


# === Baseline for comparison ===
class StandardInference:
    """Standard GUI-AIMA inference (single high-res pass) for comparison."""
    
    def __init__(self, model_path: str, device: str = "cuda", max_pixels: int = 5720064):
        from gui_aima.inference import inference as guiaima_inference
        self._inference = guiaima_inference
        
        self.model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        ).to(device).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        self.tokenizer = self.processor.tokenizer
    
    def predict(self, image: Image.Image, instruction: str) -> Tuple[float, float]:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": grounding_system_message}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ]},
        ]
        pred = self._inference(
            conversation, self.model, self.tokenizer, self.processor,
            use_placeholder=True, topk=3,
        )
        if pred["topk_points"]:
            return pred["topk_points"][0]
        return 0.5, 0.5


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--num_rounds", type=int, default=3)
    args = parser.parse_args()
    
    config = FoveationConfig(num_rounds=args.num_rounds)
    fov = FoveatedInference(args.model_path, config)
    
    px, py = fov.predict(args.image, args.instruction)
    print(f"\nFinal prediction: ({px:.4f}, {py:.4f})")
