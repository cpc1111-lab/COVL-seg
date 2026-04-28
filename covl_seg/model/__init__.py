"""Model components for COVL-Seg."""

from .boundary_detect import BoundaryDetector
from .continual_backbone import ContinualBackbone
from .covl_seg_model import COVLSegModel
from .covl_seg_model_new import COVLSegModelV2
from .dino_extractor import DINOv2FeatureExtractor
from .fusion import FusionHead
from .fusion_head import ContinualFusionHead
from .hciba_head import HCIBAHead
from .hciba_multi_scale_head import HCIBAMultiScaleHead

__all__ = [
    "BoundaryDetector",
    "ContinualBackbone",
    "COVLSegModel",
    "COVLSegModelV2",
    "DINOv2FeatureExtractor",
    "ContinualFusionHead",
    "FusionHead",
    "HCIBAHead",
    "HCIBAMultiScaleHead",
]
