from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch.nn import functional as F


def _as_index_tensor(class_indexes: object, *, device: torch.device) -> torch.Tensor:
    if class_indexes is None:
        return torch.empty(0, dtype=torch.long, device=device)
    if isinstance(class_indexes, torch.Tensor):
        return class_indexes.to(device=device, dtype=torch.long).flatten()
    if isinstance(class_indexes, Sequence):
        if len(class_indexes) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.as_tensor(class_indexes, dtype=torch.long, device=device).flatten()
    if isinstance(class_indexes, Iterable):
        values = list(class_indexes)
        if len(values) == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.as_tensor(values, dtype=torch.long, device=device).flatten()
    raise TypeError(f"Unsupported class_indexes type: {type(class_indexes)!r}")


def zero_loss(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device, dtype=torch.float32)


def kd_loss_on_class_indexes(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    class_indexes: object,
    temperature: float,
) -> torch.Tensor:
    indexes = _as_index_tensor(class_indexes, device=student_logits.device)
    if indexes.numel() == 0:
        return zero_loss(student_logits.device)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    student_selected = student_logits.index_select(dim=1, index=indexes)
    teacher_selected = teacher_logits.index_select(dim=1, index=indexes)

    student_flat = student_selected.movedim(1, -1).reshape(-1, student_selected.shape[1])
    teacher_flat = teacher_selected.movedim(1, -1).reshape(-1, teacher_selected.shape[1])
    if student_flat.shape[0] == 0:
        return zero_loss(student_logits.device)

    scaled_student = student_flat / temperature
    scaled_teacher = teacher_flat / temperature
    student_log_probs = F.log_softmax(scaled_student, dim=1)
    teacher_probs = F.softmax(scaled_teacher, dim=1)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return loss * (temperature ** 2)
