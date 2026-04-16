from typing import Dict

import torch


def compute_basic_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        tgt_mask = target == class_id
        inter = (pred_mask & tgt_mask).sum().item()
        union = (pred_mask | tgt_mask).sum().item()
        if union > 0:
            ious.append(inter / union)
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious) * 100.0)


def summarize_metrics(miou_all: float, miou_old: float, miou_new: float, bg_miou: float) -> Dict[str, float]:
    return {
        "mIoU_all": float(miou_all),
        "mIoU_old": float(miou_old),
        "mIoU_new": float(miou_new),
        "BG-mIoU": float(bg_miou),
    }
