import torch
from torch import nn
import torch.nn.functional as F


class ContinualFusionHead(nn.Module):
    """Parameterized fusion head with learnable alpha/tau for continual learning.

    Alpha controls the balance between backbone and CLIP streams (sigmoid-activated).
    Tau controls the softmax temperature (softplus + 0.1 for numerical stability).
    """

    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.tau = nn.Parameter(torch.tensor(1.0))
        self.text_proj = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(
        self,
        backbone_logits: torch.Tensor,
        clip_logits: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        alpha = torch.sigmoid(self.alpha)
        tau = F.softplus(self.tau) + 0.1

        if clip_logits.shape[-2:] != backbone_logits.shape[-2:]:
            clip_logits = F.interpolate(
                clip_logits,
                size=backbone_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        p_backbone = F.softmax(backbone_logits / tau, dim=1)
        p_clip = F.softmax(clip_logits, dim=1)

        fused = alpha * p_backbone + (1 - alpha) * p_clip
        fused = fused.clamp(min=1e-6)
        output = torch.log(fused) * tau

        return output