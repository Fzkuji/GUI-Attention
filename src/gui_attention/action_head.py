"""ActionHead: self-attention + MLP_V + MLP_T + scaled dot product + soft-argmax.

Modeled after GUI-Actor's VisionHead_MultiPatch:
  - Self-attention over visual tokens for cross-token context
  - LayerNorm + residual connection
  - MLP projections for visual and anchor tokens
  - Scaled dot-product attention for localization
  - Soft-argmax with learnable β for sub-patch coordinate prediction
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
        attn_weights = softmax(logits * β, dim=-1)
        (pred_x, pred_y) = soft_argmax(attn_weights, grid_coords)
    """

    def __init__(self, d_model: int, projection_dim: int = None,
                 num_attention_heads: int = 8, dropout_rate: float = 0.1,
                 init_beta: float = 5.0):
        super().__init__()
        if projection_dim is None:
            projection_dim = d_model
        self.d_model = d_model

        # Learnable temperature for soft-argmax
        # β controls sharpness: small → diffuse, large → approaching argmax
        self.beta = nn.Parameter(torch.tensor(init_beta))

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

    def _compute_logits(self, visual_hidden, anchor_hidden, mask=None):
        """Compute raw logits from visual and anchor hidden states.

        Returns:
            logits: (n_anchor, n_vis) raw scores.
            visual_ctx: (n_vis, d_model) contextualized visual features.
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

        return logits

    def soft_argmax(self, logits, n_height, n_width, image_offset=0, n_image_tokens=None):
        """Differentiable soft-argmax over a spatial grid of tokens.

        Args:
            logits: (n_anchor, n_vis) raw logits (before softmax).
            n_height: grid height of target image.
            n_width: grid width of target image.
            image_offset: start index of target image tokens in the n_vis dim.
            n_image_tokens: number of tokens in target image (default: n_height * n_width).

        Returns:
            pred_x: (n_anchor,) predicted x in [0, 1].
            pred_y: (n_anchor,) predicted y in [0, 1].
        """
        if n_image_tokens is None:
            n_image_tokens = n_height * n_width

        # Extract logits for the target image region
        img_logits = logits[:, image_offset:image_offset + n_image_tokens]  # (n_anchor, H*W)

        # Apply learnable temperature and softmax
        # Clamp beta to avoid numerical issues (min 0.1, max 50)
        beta = self.beta.clamp(min=0.1, max=50.0)
        weights = F.softmax(img_logits * beta, dim=-1)  # (n_anchor, H*W)

        # Build coordinate grid: center of each patch
        device = logits.device
        rows = torch.arange(n_height, device=device, dtype=logits.dtype)
        cols = torch.arange(n_width, device=device, dtype=logits.dtype)
        # Normalised coordinates [0, 1] — patch centers
        y_coords = (rows + 0.5) / n_height  # (H,)
        x_coords = (cols + 0.5) / n_width   # (W,)

        # Flatten to (H*W,) matching the token order (row-major)
        grid_y = y_coords.unsqueeze(1).expand(n_height, n_width).reshape(-1)  # (H*W,)
        grid_x = x_coords.unsqueeze(0).expand(n_height, n_width).reshape(-1)  # (H*W,)

        # Weighted average — differentiable!
        pred_x = (weights * grid_x.unsqueeze(0)).sum(dim=-1)  # (n_anchor,)
        pred_y = (weights * grid_y.unsqueeze(0)).sum(dim=-1)  # (n_anchor,)

        return pred_x, pred_y

    def forward(self, visual_hidden, anchor_hidden, labels=None, mask=None,
                grid_info=None, gt_coords=None, coord_loss_weight=1.0):
        """Forward pass.

        Args:
            visual_hidden: (n_vis, d_model) visual token hidden states.
            anchor_hidden: (n_anchor, d_model) anchor/pointer token hidden states.
            labels: optional (n_anchor, n_vis) binary overlap labels.
                    Will be normalised to probability distribution internally.
            mask: optional (n_vis,) bool tensor. True = masked (set logit to -inf).
                  Used to mask low-res patches covered by high-res crop.
            grid_info: optional dict with keys:
                - "n_height": int, grid height of target image
                - "n_width": int, grid width of target image
                - "image_offset": int, start index in n_vis (default 0)
                - "n_image_tokens": int (default n_height * n_width)
                Required for soft-argmax coordinate prediction.
            gt_coords: optional (2,) tensor [gt_x, gt_y] in [0, 1].
                       If provided with grid_info, computes coord_loss (L1).
            coord_loss_weight: weight for coordinate loss (default 1.0).

        Returns:
            attn_weights: (n_anchor, n_vis) attention weights (softmax).
            loss: scalar total loss (KL + coord) if labels/gt_coords provided, else None.
            logits: (n_anchor, n_vis) raw logits before softmax.
            pred_coords: optional (2,) tensor [pred_x, pred_y] if grid_info provided, else None.
        """
        logits = self._compute_logits(visual_hidden, anchor_hidden, mask)

        attn_weights = F.softmax(logits, dim=-1)  # (n_anchor, n_vis)

        loss = None

        # KL loss (existing)
        if labels is not None:
            eps = 1e-8
            target_dist = labels.float()
            row_sums = target_dist.sum(dim=-1, keepdim=True)
            target_dist = target_dist / (row_sums + eps)

            pred_log = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(pred_log, target_dist, reduction="batchmean")

        # Soft-argmax coordinate prediction
        pred_coords = None
        if grid_info is not None:
            pred_x, pred_y = self.soft_argmax(
                logits,
                n_height=grid_info["n_height"],
                n_width=grid_info["n_width"],
                image_offset=grid_info.get("image_offset", 0),
                n_image_tokens=grid_info.get("n_image_tokens", None),
            )
            pred_coords = torch.stack([pred_x.squeeze(), pred_y.squeeze()])  # (2,)

            # Coordinate L1 loss
            if gt_coords is not None:
                gt = gt_coords.to(logits.device).float()
                coord_loss = F.l1_loss(pred_coords, gt)
                if loss is not None:
                    loss = loss + coord_loss_weight * coord_loss
                else:
                    loss = coord_loss_weight * coord_loss

        return attn_weights, loss, logits, pred_coords
