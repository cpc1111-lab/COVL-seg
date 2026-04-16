"""Model components for COVL-Seg."""

from .boundary_detect import BoundaryDetector
from .continual_backbone import ContinualBackbone
from .covl_seg_model import COVLSegModel
from .fusion import FusionHead
from .hciba_head import HCIBAHead

__all__ = [
    "BoundaryDetector",
    "ContinualBackbone",
    "COVLSegModel",
    "FusionHead",
    "HCIBAHead",
]
