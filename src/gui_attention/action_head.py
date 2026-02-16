"""ActionHead: MLP_V + MLP_T + scaled dot product + KL loss.

Simplified from GUI-Actor's VisionHead_MultiPatch by removing self-attention,
since LLM last-layer hidden states already have cross-token context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    """Action head that predicts click location via attention over visual patches.

    Architecture:
        visual_hidden → MLP_V → proj_v   (n_vis, d_model)
        anchor_hidden → MLP_T → proj_t   (n_anchor, d_model)
        logits = (proj_t @ proj_v^T) / sqrt(d_model)   (n_anchor, n_vis)
        attn_weights = softmax(logits, dim=-1)

    No self-attention layer (unlike GUI-Actor), because we use LLM last-layer
    hidden states which already incorporate cross-token context.
    """

    def __init__(self, d_model: int, projection_dim: int = None):
        super().__init__()
        if projection_dim is None:
            projection_dim = d_model
        self.d_model = d_model

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
        """
        proj_v = self.mlp_v(visual_hidden)   # (n_vis, d_model)
        proj_t = self.mlp_t(anchor_hidden)   # (n_anchor, d_model)

        # Scaled dot product
        scale = self.d_model ** 0.5
        logits = torch.matmul(proj_t, proj_v.t()) / scale  # (n_anchor, n_vis)

        # Apply mask: set logits to large negative value for masked patches.
        # Use -1e9 instead of -inf to avoid NaN in KL computation
        # (0 * log_softmax(-inf) = 0 * (-inf) = NaN).
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

        return attn_weights, loss
