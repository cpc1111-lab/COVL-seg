from cat_seg import (
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
)
from cat_seg import CATSeg
from cat_seg.config import add_cat_seg_config


def add_covl_seg_d2_config(cfg):
    add_cat_seg_config(cfg)


def add_mask_former_config(cfg):
    add_covl_seg_d2_config(cfg)
