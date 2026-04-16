import torch
from torch import nn


class FusionHead(nn.Module):
    """Fuses continual and CLIP streams with log-space product-of-experts."""

    def __init__(self, alpha: float = 0.5, tau: float = 1.0, boundary_boost: float = 0.25):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.boundary_boost = boundary_boost

    def forward(
        self,
        seg_logits: torch.Tensor,
        clip_logits: torch.Tensor,
        boundary_map: torch.Tensor,
    ) -> torch.Tensor:
        if clip_logits.shape[-2:] != seg_logits.shape[-2:]:
            clip_logits = nn.functional.interpolate(
                clip_logits,
                size=seg_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if boundary_map.shape[-2:] != seg_logits.shape[-2:]:
            boundary_map = nn.functional.interpolate(
                boundary_map,
                size=seg_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        alpha_map = self.alpha + self.boundary_boost * boundary_map
        alpha_map = alpha_map.clamp(0.0, 1.0)

        seg_log_prob = nn.functional.log_softmax(seg_logits, dim=1)
        clip_log_prob = nn.functional.log_softmax(clip_logits, dim=1)
        fused = (alpha_map * seg_log_prob + (1.0 - alpha_map) * clip_log_prob) / max(self.tau, 1e-6)
        return fused
