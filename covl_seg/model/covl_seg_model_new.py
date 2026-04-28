"""COVLSegModelV2: integrates CLIP + DINOv2 + HCIBA + ContinualFusion for
open-vocabulary segmentation with continual learning support."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .clip_encoder import CLIPTextEncoder, CLIPVisualEncoder
from .dino_extractor import DINOv2FeatureExtractor
from .fusion_head import ContinualFusionHead
from .hciba_multi_scale_head import HCIBAMultiScaleHead

_DINO_DIMS = {
    "dinov2_vits14": (384, 384, 384),
    "dinov2_vitb14": (768, 768, 768),
    "dinov2_vitl14": (1024, 1024, 1024),
}


class COVLSegModelV2(nn.Module):
    """Open-vocabulary segmentation model fusing CLIP, DINOv2, HCIBA, and
    continual-fusion heads."""

    def __init__(
        self,
        clip_model_name: str = "ViT-B-16",
        dino_model_name: str = "dinov2_vitb14",
        clip_finetune: str = "none",
        num_classes: int = 150,
        out_dim: int = 768,
    ):
        super().__init__()
        self.clip_visual = CLIPVisualEncoder(
            model_name=clip_model_name, clip_finetune=clip_finetune
        )
        self.clip_text = CLIPTextEncoder(
            model_name=clip_model_name, output_dim=out_dim
        )
        self.dino = DINOv2FeatureExtractor(model_name=dino_model_name)

        clip_dim = self.clip_visual.dim
        dino_dims = self._get_dino_dims(dino_model_name)

        self.hciba_head = HCIBAMultiScaleHead(
            clip_dim=clip_dim, dino_dims=dino_dims, out_dim=out_dim
        )
        self.fusion_head = ContinualFusionHead(
            num_classes=num_classes, feature_dim=out_dim
        )
        self.clip_logit_proj = nn.Linear(clip_dim, out_dim, bias=False)

        self.num_classes = num_classes
        self.out_dim = out_dim

    @staticmethod
    def _get_dino_dims(model_name: str) -> Tuple[int, int, int]:
        return _DINO_DIMS.get(model_name, (768, 768, 768))

    def forward(
        self,
        images: torch.Tensor,
        class_names: List[str],
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        text_embeddings = self.clip_text(class_names)
        clip_dense = self.clip_visual.get_dense_features(images)
        dino_features = self.dino(images)
        projected = self.hciba_head(clip_dense, dino_features)

        backbone_logits = torch.einsum(
            "bchw,kc->bkhw",
            F.normalize(projected, dim=1),
            F.normalize(text_embeddings, dim=1),
        )

        clip_cls = clip_dense[:, 0, :]
        clip_cls_proj = self.clip_logit_proj(clip_cls)
        clip_logits_flat = torch.einsum(
            "bd,kd->bk",
            F.normalize(clip_cls_proj, dim=-1),
            F.normalize(text_embeddings, dim=-1),
        )
        H, W = projected.shape[-2:]
        clip_logits = clip_logits_flat.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, H, W
        )

        fused_logits = self.fusion_head(backbone_logits, clip_logits, text_embeddings)

        outputs: Dict[str, torch.Tensor] = {"logits": fused_logits}
        if targets is not None:
            loss = F.cross_entropy(fused_logits, targets, ignore_index=255)
            outputs["loss"] = loss
        return outputs

    def inject_alpha_tau(self, alpha: float, tau: float) -> None:
        self.fusion_head.alpha.data.fill_(alpha)
        self.fusion_head.tau.data.fill_(tau)