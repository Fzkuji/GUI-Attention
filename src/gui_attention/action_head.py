"""ActionHead: self-attention + MLP_V + MLP_T + scaled dot product + KL loss.

Modeled after GUI-Actor's VisionHead_MultiPatch:
  - Self-attention over visual tokens for cross-token context
  - LayerNorm + residual connection
  - MLP projections for visual and anchor tokens
  - Scaled dot-product attention for localization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    """Action head that predicts click location via attention over visual patches.

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

        # Self-attention for visual tokens (like GUI-Actor)
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
            visual_hidden: (n_vis, d_model) visual token hidden states.
            anchor_hidden: (n_anchor, d_model) anchor/pointer token hidden states.
            labels: optional (n_anchor, n_vis) binary overlap labels.
                    Will be normalised to probability distribution internally.
            mask: optional (n_vis,) bool tensor. True = masked (set logit to -inf).
                  Used to mask low-res patches covered by high-res crop.

        Returns:
            attn_weights: (n_anchor, n_vis) attention weights (softmax).
            loss: scalar KL divergence loss if labels provided, else None.
            logits: (n_anchor, n_vis) raw logits before softmax.
        """
        # Self-attention over visual tokens (add batch dim)
        enc_input = visual_hidden.unsqueeze(0)  # (1, n_vis, d_model)
        attn_output, _ = self.self_attention(
            query=enc_input, key=enc_input, value=enc_input,
            need_weights=False,
        )
        # Residual + LayerNorm
        visual_ctx = self.layer_norm(enc_input + self.dropout(attn_output))
        visual_ctx = visual_ctx.squeeze(0)  # (n_vis, d_model)

        # Project
        proj_v = self.mlp_v(visual_ctx)    # (n_vis, d_model)
        proj_t = self.mlp_t(anchor_hidden) # (n_anchor, d_model)

        # Scaled dot product
        scale = self.d_model ** 0.5
        logits = torch.matmul(proj_t, proj_v.t()) / scale  # (n_anchor, n_vis)

        # Apply mask
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(0), -1e9)

        attn_weights = F.softmax(logits, dim=-1)  # (n_anchor, n_vis)

        loss = None
        if labels is not None:
            eps = 1e-8
            target_dist = labels.float()
            row_sums = target_dist.sum(dim=-1, keepdim=True)
            target_dist = target_dist / (row_sums + eps)

            pred_log = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(pred_log, target_dist, reduction="batchmean")

        return attn_weights, loss, logits
