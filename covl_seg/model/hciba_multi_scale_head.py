from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class HCIBAMultiScaleHead(nn.Module):
    """Multi-scale cross-attention head fusing CLIP and DINOv2 features."""

    SCALES = ("res3", "res4", "res5")

    def __init__(
        self,
        clip_dim: int = 768,
        dino_dims: Tuple[int, int, int] = (384, 384, 768),
        out_dim: int = 768,
        num_heads: int = 8,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.dino_projections = nn.ModuleDict(
            {
                scale: nn.Conv2d(dim, out_dim, kernel_size=1)
                for scale, dim in zip(self.SCALES, dino_dims)
            }
        )

        self.q_proj = nn.Linear(clip_dim, out_dim)
        self.k_proj = nn.Linear(out_dim, out_dim)
        self.v_proj = nn.Linear(out_dim, out_dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim * 3, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        clip_features: torch.Tensor,
        dino_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B = clip_features.shape[0]
        head_dim = self.out_dim // self.num_heads

        q = self.q_proj(clip_features[:, 0, :]).unsqueeze(1)
        q_h = q.view(B, 1, self.num_heads, head_dim).transpose(1, 2)

        target_size = dino_features["res3"].shape[-2:]
        scale_outputs = []

        for scale in self.SCALES:
            feat = self.dino_projections[scale](dino_features[scale])
            _, _, H_i, W_i = feat.shape
            feat_flat = feat.flatten(2).permute(0, 2, 1)

            k = self.k_proj(feat_flat)
            v = self.v_proj(feat_flat)

            k_h = k.view(B, -1, self.num_heads, head_dim).transpose(1, 2)
            v_h = v.view(B, -1, self.num_heads, head_dim).transpose(1, 2)

            attn = (q_h @ k_h.transpose(-2, -1)) / (head_dim**0.5)
            attn = F.softmax(attn, dim=-1)
            attn_out = (attn @ v_h).transpose(1, 2).reshape(B, self.out_dim)

            modulated = feat * attn_out.view(B, self.out_dim, 1, 1)

            if modulated.shape[-2:] != target_size:
                modulated = F.interpolate(
                    modulated, size=target_size, mode="bilinear", align_corners=False
                )

            scale_outputs.append(modulated)

        fused = torch.cat(scale_outputs, dim=1)
        return self.fusion(fused)