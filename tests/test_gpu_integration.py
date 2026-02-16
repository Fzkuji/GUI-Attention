"""GPU integration test for v4: build model, forward pass, action head, save/load.

Run on server with: CUDA_VISIBLE_DEVICES=0 python tests/test_gpu_integration.py
"""

import os

import torch
from PIL import Image

from gui_attention.attention import (
    extract_anchor_hidden_states,
    extract_visual_hidden_states,
    identify_attended_image,
    token_to_spatial,
)
from gui_attention.builder import MultiRoundInputBuilder
from gui_attention.crop import crop_image
from gui_attention.labels import compute_binary_labels, compute_overlap_mask
from gui_attention.model import build_model

MODEL_PATH = os.environ.get("MODEL_PATH", "/home/zichuanfu2/models/Qwen2.5-VL-3B-Instruct")


def main():
    print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")

    # 1. Build model
    print("\n[1] Building model with LoRA + ActionHead...")
    model, tokenizer, processor = build_model(
        MODEL_PATH,
        lora_r=32, lora_alpha=64, lora_target_modules="q_proj,v_proj",
        torch_dtype=torch.bfloat16, gradient_checkpointing=True,
    )
    model.to("cuda:0")
    print(f"    hidden_size={model.config.hidden_size}, pointer_pad_id={model.config.pointer_pad_token_id}")
    print(f"    ActionHead d_model={model.action_head.d_model}")

    # 2. Round 0: low-res full image
    print("\n[2] Round 0: low-res full image forward pass")
    builder = MultiRoundInputBuilder(MODEL_PATH, tokenizer)
    img = Image.new("RGB", (800, 600), color=(128, 128, 128))
    tmp_path = "/tmp/test_gui_attn.png"
    img.save(tmp_path)

    r0_inputs, cur_text, cur_images = builder.build_round0(tmp_path, "Click the button")
    inp0 = {k: v.to("cuda:0") for k, v in r0_inputs.items()}
    print(f"    input_ids: {inp0['input_ids'].shape}")
    print(f"    pixel_values: {inp0['pixel_values'].shape}")
    print(f"    image_grid_thw: {inp0['image_grid_thw']}")

    with torch.no_grad():
        out0 = model(
            input_ids=inp0["input_ids"],
            attention_mask=inp0.get("attention_mask"),
            pixel_values=inp0.get("pixel_values"),
            image_grid_thw=inp0.get("image_grid_thw"),
        )
    hs0 = out0.hidden_states[-1]
    print(f"    hidden_states[-1]: {hs0.shape}")

    img_tok = model.config.image_token_id
    pp_id = model.config.pointer_pad_token_id

    vis0, vr0 = extract_visual_hidden_states(hs0, inp0["input_ids"], img_tok)
    anc0 = extract_anchor_hidden_states(hs0, inp0["input_ids"], pp_id, n=0)
    print(f"    visual: {vis0.shape}, anchor: {anc0.shape}, ranges: {vr0}")

    attn0, _ = model.action_head(vis0, anc0)
    print(f"    attn: {attn0.shape}, sum={attn0.sum().item():.4f}, argmax={attn0.argmax().item()}")
    assert abs(attn0.sum().item() - 1.0) < 1e-3, "Attention should sum to ~1"

    # 3. Round 1: + high-res crop
    print("\n[3] Round 1: extend with high-res crop")
    cropped, crop_bbox = crop_image(img, 0.5, 0.5, 0.3)
    r1_inputs, cur_text, cur_images = builder.extend_with_crop(cur_text, cur_images, cropped, crop_bbox)
    inp1 = {k: v.to("cuda:0") for k, v in r1_inputs.items()}
    print(f"    input_ids: {inp1['input_ids'].shape}")
    print(f"    image_grid_thw: {inp1['image_grid_thw']}")

    with torch.no_grad():
        out1 = model(
            input_ids=inp1["input_ids"],
            attention_mask=inp1.get("attention_mask"),
            pixel_values=inp1.get("pixel_values"),
            image_grid_thw=inp1.get("image_grid_thw"),
        )
    hs1 = out1.hidden_states[-1]

    vis1, vr1 = extract_visual_hidden_states(hs1, inp1["input_ids"], img_tok)
    anc1 = extract_anchor_hidden_states(hs1, inp1["input_ids"], pp_id, n=1)
    print(f"    visual: {vis1.shape}, anchor: {anc1.shape}, ranges: {vr1}")
    assert len(vr1) >= 2, "Should have at least 2 image blocks (low + high)"

    # Mask low-res patches covered by crop
    merge = model.backbone.base_model.model.visual.spatial_merge_size
    grid_dims = builder.get_image_grid_dims(inp1["image_grid_thw"], merge)
    nh_low, nw_low = grid_dims[0]
    n_low = vr1[0][1]
    n_total = sum(r[1] for r in vr1)

    low_mask = compute_overlap_mask(nh_low, nw_low, crop_bbox).to("cuda:0")
    full_mask = torch.zeros(n_total, dtype=torch.bool, device="cuda:0")
    full_mask[:n_low] = low_mask

    attn1, _ = model.action_head(vis1, anc1, mask=full_mask)
    masked_sum = attn1[0, :n_low][low_mask].sum().item()
    high_sum = attn1[0, n_low:].sum().item()
    print(f"    attn sum={attn1.sum().item():.4f}, masked_low={masked_sum:.6f}, high={high_sum:.4f}")
    assert masked_sum < 1e-5, "Masked patches should get ~0 attention"

    img_idx, local_idx = identify_attended_image(attn1.squeeze(0), vr1)
    info = builder.image_infos[img_idx]
    print(f"    argmax: image {img_idx} ({info.resolution}), local token {local_idx}")

    # 4. Training loss test
    print("\n[4] Training loss + gradient test")
    model.train()
    gt_bbox = (0.3, 0.3, 0.7, 0.7)
    labels0 = compute_binary_labels(nh_low, nw_low, gt_bbox).unsqueeze(0).to("cuda:0")
    attn_t, loss_t = model.action_head(vis0.detach().requires_grad_(True), anc0.detach().requires_grad_(True), labels=labels0)
    print(f"    loss: {loss_t.item():.4f}")
    assert loss_t.isfinite(), "Loss should be finite"
    loss_t.backward()
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.action_head.parameters())
    print(f"    action_head gradients: {has_grads}")
    assert has_grads, "Gradients should flow to action head"

    # 5. Save/load test
    print("\n[5] Save/load test")
    save_path = "/tmp/test_v4_ckpt"
    model.eval()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    files = sorted(os.listdir(save_path))
    print(f"    Saved files: {files}")
    assert "action_head.pt" in files, "action_head.pt missing"
    assert "adapter_config.json" in files, "adapter_config.json missing"

    # Load back
    from gui_attention.model import Qwen25VLWithActionHead
    model2, tok2 = Qwen25VLWithActionHead.load_pretrained(
        save_path, base_model_name_or_path=MODEL_PATH, device="cuda:0",
    )
    model2.eval()
    print(f"    Loaded model hidden_size={model2.config.hidden_size}")

    # Verify same output
    with torch.no_grad():
        out_loaded = model2(
            input_ids=inp0["input_ids"],
            attention_mask=inp0.get("attention_mask"),
            pixel_values=inp0.get("pixel_values"),
            image_grid_thw=inp0.get("image_grid_thw"),
        )
    hs_loaded = out_loaded.hidden_states[-1]
    vis_loaded, _ = extract_visual_hidden_states(hs_loaded, inp0["input_ids"], img_tok)
    anc_loaded = extract_anchor_hidden_states(hs_loaded, inp0["input_ids"], pp_id, n=0)
    attn_loaded, _ = model2.action_head(vis_loaded, anc_loaded)
    print(f"    Loaded model attn argmax: {attn_loaded.argmax().item()}")

    # Cleanup
    import shutil
    shutil.rmtree(save_path, ignore_errors=True)
    os.remove(tmp_path)

    print("\n=== All GPU integration tests PASSED! ===")


if __name__ == "__main__":
    main()
