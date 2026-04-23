from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch


def load_visible_class_indexes(path_value: str, num_classes: int) -> Optional[torch.Tensor]:
    path_str = str(path_value).strip() if path_value is not None else ""
    if not path_str:
        return None

    candidate = Path(path_str)
    if not candidate.exists():
        return None

    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, list):
        return None

    indexes = []
    for item in payload:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < int(num_classes):
            indexes.append(idx)

    if not indexes:
        return None

    return torch.tensor(sorted(set(indexes)), dtype=torch.long)


def select_class_channels(tensor: torch.Tensor, class_indexes: Optional[torch.Tensor]) -> torch.Tensor:
    if class_indexes is None or class_indexes.numel() == 0:
        return tensor
    indexes = class_indexes.to(device=tensor.device, dtype=torch.long)
    return tensor.index_select(-1, indexes)


def mask_logits_and_targets_to_visible_classes(
    logits: torch.Tensor,
    targets: torch.Tensor,
    visible_class_indexes: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if visible_class_indexes is None or visible_class_indexes.numel() == 0:
        return logits, targets
    return select_class_channels(logits, visible_class_indexes), select_class_channels(
        targets, visible_class_indexes
    )
