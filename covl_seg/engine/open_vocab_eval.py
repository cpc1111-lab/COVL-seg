from typing import Dict, Mapping, Optional

import torch

from .evaluator import compute_basic_miou


class OpenVocabEvaluator:
    """Minimal open-vocabulary evaluator scaffold.

    This evaluator accepts class-logits and target labels and reports
    `ov_mIoU_<dataset_key>` style metrics.
    """

    def __init__(self, dataset_aliases: Optional[Mapping[str, str]] = None):
        self.dataset_aliases = dict(dataset_aliases or {})

    def evaluate_dataset(
        self,
        dataset_key: str,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        if logits.ndim != 4:
            raise ValueError("logits must be [B, C, H, W]")
        if targets.ndim != 3:
            raise ValueError("targets must be [B, H, W]")

        pred = logits.argmax(dim=1)
        num_classes = logits.shape[1]
        miou = compute_basic_miou(pred=pred, target=targets, num_classes=num_classes)
        return {f"ov_mIoU_{dataset_key}": float(miou)}

    def evaluate_all(self, dataset_outputs: Mapping[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for key, pack in dataset_outputs.items():
            merged.update(
                self.evaluate_dataset(
                    dataset_key=key,
                    logits=pack["logits"],
                    targets=pack["targets"],
                )
            )
        return merged
