from typing import Dict, Optional

import torch
from torch import nn

from .boundary_detect import BoundaryDetector
from .continual_backbone import ContinualBackbone
from .fusion import FusionHead
from .hciba_head import HCIBAHead


class COVLSegModel(nn.Module):
    """Minimal COVL-Seg model composition for iterative development."""

    def __init__(
        self,
        backbone: ContinualBackbone,
        hciba_head: HCIBAHead,
        boundary_detector: BoundaryDetector,
        fusion_head: FusionHead,
        num_classes: int,
        text_dim: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.hciba_head = hciba_head
        self.boundary_detector = boundary_detector
        self.fusion_head = fusion_head
        self.num_classes = num_classes
        self.text_dim = text_dim

    def _compute_seg_logits(self, projected: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        visual = nn.functional.normalize(projected, dim=1)
        text = nn.functional.normalize(text_embeddings, dim=1)
        return torch.einsum("bchw,kc->bkhw", visual, text)

    def forward(
        self,
        images: torch.Tensor,
        text_embeddings: torch.Tensor,
        clip_logits: Optional[torch.Tensor] = None,
        clip_attention_map: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if text_embeddings.shape[0] != self.num_classes:
            raise ValueError("text_embeddings class count must match num_classes")
        if text_embeddings.shape[1] != self.text_dim:
            raise ValueError("text_embeddings feature dimension must match text_dim")

        features = self.backbone(images, out_size=images.shape[-2:])
        boundary_map = self.boundary_detector(images=images, attention_map=clip_attention_map)
        projected = self.hciba_head(features, boundary_map)
        logits = self._compute_seg_logits(projected, text_embeddings)

        if clip_logits is not None:
            logits = self.fusion_head(logits, clip_logits, boundary_map)

        outputs: Dict[str, torch.Tensor] = {
            "logits": logits,
            "boundary_map": boundary_map,
        }
        if targets is not None:
            outputs["loss"] = nn.functional.cross_entropy(logits, targets)
        return outputs
