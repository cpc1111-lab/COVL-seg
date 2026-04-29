"""Data modules for COVL-Seg."""

from .ade20k_15 import split_path as ade20k_15_split_path
from .ade20k_100 import split_path as ade20k_100_split_path
from .coco_stuff_164_10 import split_path as coco_stuff_164_10_split_path
from .continual_loader import build_mixed_batch, validate_split_mapping
from .datasets import ADE20KDataset, COCOStuffDataset, SegmentationAugmentation, SegmentationEvalTransform
from .download import ensure_dataset, ensure_ade20k, ensure_coco_stuff
from .text_embedding_db import TextEmbeddingDB

__all__ = [
    "ADE20KDataset",
    "COCOStuffDataset",
    "SegmentationAugmentation",
    "SegmentationEvalTransform",
    "TextEmbeddingDB",
    "ade20k_15_split_path",
    "ade20k_100_split_path",
    "build_mixed_batch",
    "coco_stuff_164_10_split_path",
    "ensure_dataset",
    "ensure_ade20k",
    "ensure_coco_stuff",
    "validate_split_mapping",
]
