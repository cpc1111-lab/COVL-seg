from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_runtime_continual_losses_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "covl_seg_d2_runtime"
        / "cat_seg"
        / "continual_losses.py"
    )
    spec = importlib.util.spec_from_file_location("runtime_continual_losses", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kd_loss_on_class_indexes_is_finite_and_non_negative_for_valid_indexes():
    module = _load_runtime_continual_losses_module()

    student_logits = torch.randn(2, 5, 3, 3)
    teacher_logits = torch.randn(2, 5, 3, 3)
    class_indexes = [0, 2, 4]

    loss = module.kd_loss_on_class_indexes(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        class_indexes=class_indexes,
        temperature=2.0,
    )

    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_kd_loss_on_class_indexes_returns_zero_for_empty_indexes():
    module = _load_runtime_continual_losses_module()

    student_logits = torch.randn(1, 4, 2, 2)
    teacher_logits = torch.randn(1, 4, 2, 2)

    loss = module.kd_loss_on_class_indexes(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        class_indexes=[],
        temperature=1.0,
    )

    assert float(loss.item()) == 0.0


def test_continual_losses_module_exposes_kd_api():
    module = _load_runtime_continual_losses_module()

    assert hasattr(module, "kd_loss_on_class_indexes")
    assert callable(module.kd_loss_on_class_indexes)
