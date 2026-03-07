"""Dual Head architecture: LookHead (explore) + ClickHead (precise click).

LookHead selects which region to crop (coarse localization on full image).
ClickHead selects precise click position within a high-res crop.
Two independent heads with separate parameters — no interference.

Training:
  Phase 1: Only LookHead trains (learn to select good crop regions).
  Phase 2: Both heads train (LookHead selects crop, ClickHead clicks on it).

Inference:
  Round 0: LookHead on low-res → select crop center.
  Round N: LookHead on low-res + crops.
    If LookHead attends high-res crop → ClickHead on that crop → precise (x,y) → stop.
    If LookHead attends low-res → saccade continues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AttentionHead(nn.Module):
    """Base attention head: SelfAttn over visual + cross-attn with anchor.

    Shared architecture for both LookHead and ClickHead, but each
    instantiates its own parameters.

    Architecture:
        visual_hidden → SelfAttention → LayerNorm+Residual → MLP_V → proj_v
        anchor_hidden → MLP_T → proj_t
        logits = (proj_t @ proj_v^T) / sqrt(d_model)
        attn_weights = softmax(logits, dim=-1)
    """

    def __init__(self, d_model: int, projection_dim: int = None,
                 num_attention_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        if projection_dim is None:
            projection_dim = d_model
        self.d_model = d_model

        # Self-attention for visual tokens
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

        # MLP_V: project visual hidden states
        self.mlp_v = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model),
        )
        # MLP_T: project anchor (target/pointer) hidden states
        self.mlp_t = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, d_model),
        )

    def forward(self, visual_hidden, anchor_hidden, labels=None, mask=None):
        """Forward pass.

        Args:
            visual_hidden: (n_vis, d_model) visual token embeddings (pre-LLM ViT output).
            anchor_hidden: (n_anchor, d_model) anchor/pointer token embeddings (post-LLM).
            labels: optional (n_anchor, n_vis) target distribution for KL loss.
            mask: optional (n_vis,) bool tensor. True = masked (logit → -inf).

        Returns:
            attn_weights: (n_anchor, n_vis) attention weights.
            loss: scalar KL loss if labels provided, else None.
            logits: (n_anchor, n_vis) raw logits.
        """
        # Self-attention over visual tokens
        enc_input = visual_hidden.unsqueeze(0)
        attn_output, _ = self.self_attention(
            query=enc_input, key=enc_input, value=enc_input,
            need_weights=False,
        )
        visual_ctx = self.layer_norm(enc_input + self.dropout(attn_output))
        visual_ctx = visual_ctx.squeeze(0)

        # Project
        proj_v = self.mlp_v(visual_ctx)
        proj_t = self.mlp_t(anchor_hidden)

        # Scaled dot product
        scale = self.d_model ** 0.5
        logits = torch.matmul(proj_t, proj_v.t()) / scale

        # Apply mask
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(0), -1e9)

        attn_weights = F.softmax(logits, dim=-1)

        # KL loss
        loss = None
        if labels is not None:
            eps = 1e-8
            target_dist = labels.float()
            row_sums = target_dist.sum(dim=-1, keepdim=True)
            target_dist = target_dist / (row_sums + eps)

            pred_log = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(pred_log, target_dist, reduction="batchmean")

        return attn_weights, loss, logits


class DualActionHead(nn.Module):
    """Dual head: LookHead for exploration + ClickHead for precise clicking.

    Both heads share the same _AttentionHead architecture but have
    completely independent parameters.
    """

    def __init__(self, d_model: int, projection_dim: int = None,
                 num_attention_heads: int = 8, dropout_rate: float = 0.1):
        super().__init__()
        self.look_head = _AttentionHead(
            d_model, projection_dim, num_attention_heads, dropout_rate)
        self.click_head = _AttentionHead(
            d_model, projection_dim, num_attention_heads, dropout_rate)

    def look(self, visual_hidden, anchor_hidden, labels=None, mask=None):
        """LookHead forward: decide where to crop next.

        Returns: (attn_weights, loss, logits)
        """
        return self.look_head(visual_hidden, anchor_hidden, labels, mask)

    def click(self, visual_hidden, anchor_hidden, labels=None, mask=None):
        """ClickHead forward: decide precise click position on high-res crop.

        Returns: (attn_weights, loss, logits)
        """
        return self.click_head(visual_hidden, anchor_hidden, labels, mask)
