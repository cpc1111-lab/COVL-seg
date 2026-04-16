from typing import Tuple

import torch
from torch import nn


class HCIBAHead(nn.Module):
    """Dual projection head for boundary/semantic feature alignment."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.phi_bnd = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.phi_sem = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def project(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.phi_bnd(features), self.phi_sem(features)

    def forward(self, features: torch.Tensor, boundary_map: torch.Tensor) -> torch.Tensor:
        bnd_proj, sem_proj = self.project(features)
        if boundary_map.shape[-2:] != bnd_proj.shape[-2:]:
            boundary_map = nn.functional.interpolate(
                boundary_map,
                size=bnd_proj.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        gate = boundary_map.clamp(0.0, 1.0)
        return gate * bnd_proj + (1.0 - gate) * sem_proj
