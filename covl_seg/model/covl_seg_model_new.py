"""COVLSegModelV2: integrates CLIP + DINOv2 + HCIBA + ContinualFusion for
open-vocabulary segmentation with continual learning support."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .clip_encoder import CLIPTextEncoder, CLIPVisualEncoder, _create_or_get_clip
from .dino_extractor import DINOv2FeatureExtractor
from .fusion_head import ContinualFusionHead
from .hciba_multi_scale_head import HCIBAMultiScaleHead

_DINO_DIMS = {
    "dinov2_vits14": (384, 384, 384),
    "dinov2_vitb14": (768, 768, 768),
    "dinov2_vitl14": (1024, 1024, 1024),
}


def _check_nan(t: torch.Tensor, name: str) -> None:
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(
            f"[NaN-DEBUG] {name}: shape={tuple(t.shape)} "
            f"min={t.min().item():.4f} max={t.max().item():.4f} "
            f"mean={t.mean().item():.4f} nan={torch.isnan(t).any().item()} "
            f"inf={torch.isinf(t).any().item()}",
            flush=True,
        )


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
        shared_clip = _create_or_get_clip(clip_model_name, "openai")

        self.clip_visual = CLIPVisualEncoder(
            model_name=clip_model_name,
            pretrained="openai",
            clip_finetune=clip_finetune,
            shared_clip=shared_clip,
        )
        self.clip_text = CLIPTextEncoder(
            model_name=clip_model_name,
            pretrained="openai",
            output_dim=out_dim,
            clip_finetune="attention" if clip_finetune in ("attention", "full") else "none",
            shared_clip=shared_clip,
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

    def compute_ciba_loss(
        self,
        projected: torch.Tensor,
        boundary_map: Optional[torch.Tensor],
        mi_estimate: torch.Tensor,
        beta_star: float,
    ) -> torch.Tensor:
        """Compute CIBA alignment loss: -MI + beta_star * MSE."""
        from covl_seg.losses.ciba import ciba_alignment_loss
        if boundary_map is not None:
            target = projected * boundary_map
        else:
            target = projected
        return ciba_alignment_loss(projected, target, mi_estimate, beta_star)

    def compute_ctr_loss(
        self,
        logits: torch.Tensor,
        seen_class_ids: List[int],
        unseen_class_ids: List[int],
        gamma: float = 0.5,
    ) -> torch.Tensor:
        """Compute contrastive background loss pushing pixels away from unseen classes."""
        from covl_seg.losses.ctr import contrastive_background_loss
        return contrastive_background_loss(
            logits=logits,
            seen_class_ids=seen_class_ids,
            unseen_class_ids=unseen_class_ids,
            gamma=gamma,
        )

    def forward(
        self,
        images: torch.Tensor,
        class_names: List[str],
        targets: Optional[torch.Tensor] = None,
        seen_class_ids: Optional[List[int]] = None,
        unseen_class_ids: Optional[List[int]] = None,
        mi_estimate: Optional[torch.Tensor] = None,
        beta_star: float = 0.0,
        ciba_weight: float = 0.0,
        ctr_weight: float = 0.0,
        gamma: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        text_embeddings = self.clip_text(class_names)
        clip_dense = self.clip_visual.get_dense_features(images).float()
        text_embeddings = text_embeddings.float()
        with torch.no_grad():
            dino_features = {k: v.float() for k, v in self.dino(images).items()}
        projected = self.hciba_head(clip_dense, dino_features)

        _check_nan(projected, "projected")

        backbone_logits = torch.einsum(
            "bchw,kc->bkhw",
            F.normalize(projected.float(), dim=1),
            F.normalize(text_embeddings.float(), dim=1),
        )

        _check_nan(backbone_logits, "backbone_logits")

        clip_cls = clip_dense[:, 0, :]
        clip_cls_proj = self.clip_logit_proj(clip_cls.float())
        clip_logits_flat = torch.einsum(
            "bd,kd->bk",
            F.normalize(clip_cls_proj.float(), dim=-1),
            F.normalize(text_embeddings.float(), dim=-1),
        )
        H, W = projected.shape[-2:]
        clip_logits = clip_logits_flat.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, H, W
        )

        _check_nan(clip_logits, "clip_logits")

        fused_logits = self.fusion_head(backbone_logits.float(), clip_logits.float(), text_embeddings.float())

        _check_nan(fused_logits, "fused_logits")

        outputs: Dict[str, torch.Tensor] = {
            "logits": fused_logits,
            "projected": projected,
            "backbone_logits": backbone_logits,
            "clip_logits": clip_logits,
        }

        if targets is not None:
            logits_upsampled = F.interpolate(
                fused_logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )
            num_logits = logits_upsampled.shape[1]
            targets_safe = targets.clone()
            targets_safe[(targets_safe >= num_logits) & (targets_safe != 255)] = 255
            valid_mask = targets_safe != 255
            valid_count = valid_mask.sum().item()
            if valid_count < 1:
                seg_loss = torch.tensor(0.0, device=logits_upsampled.device, requires_grad=True)
            else:
                seg_loss = F.cross_entropy(logits_upsampled, targets_safe, ignore_index=255)
            total_loss = seg_loss

            if ciba_weight > 0.0 and mi_estimate is not None:
                ciba_loss = self.compute_ciba_loss(projected, None, mi_estimate, beta_star)
                if not (torch.isnan(ciba_loss) or torch.isinf(ciba_loss)):
                    total_loss = total_loss + ciba_weight * ciba_loss
                    outputs["loss_ciba"] = ciba_loss.detach()

            if ctr_weight > 0.0 and seen_class_ids is not None and unseen_class_ids is not None:
                ctr_loss = self.compute_ctr_loss(
                    fused_logits, seen_class_ids, unseen_class_ids, gamma
                )
                if not (torch.isnan(ctr_loss) or torch.isinf(ctr_loss)):
                    total_loss = total_loss + ctr_weight * ctr_loss
                    outputs["loss_ctr"] = ctr_loss.detach()

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[NaN-DEBUG] total_loss is NaN/Inf! seg_loss={seg_loss.item():.4f} ciba={outputs.get('loss_ciba')} ctr={outputs.get('loss_ctr')}", flush=True)
                outputs["loss"] = seg_loss
            else:
                outputs["loss"] = total_loss
            outputs["loss_seg"] = seg_loss

        return outputs

    def inject_alpha_tau(self, alpha: float, tau: float) -> None:
        self.fusion_head.alpha.data.fill_(alpha)
        self.fusion_head.tau.data.fill_(tau)