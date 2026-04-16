from typing import Optional

import torch
from torch import nn


class ContinualBackbone(nn.Module):
    """Small convolutional backbone used for shape-safe development tests."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, out_dim: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor, out_size: Optional[torch.Size] = None) -> torch.Tensor:
        features = self.stem(images)
        if out_size is not None and features.shape[-2:] != tuple(out_size):
            features = nn.functional.interpolate(
                features,
                size=out_size,
                mode="bilinear",
                align_corners=False,
            )
        return features
