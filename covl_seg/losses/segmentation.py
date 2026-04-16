import torch
from torch import nn


def masked_segmentation_ce(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    if logits.ndim != 4:
        raise ValueError("logits must be [B, C, H, W]")
    if targets.ndim != 3:
        raise ValueError("targets must be [B, H, W]")
    return nn.functional.cross_entropy(logits, targets.long(), ignore_index=ignore_index)
