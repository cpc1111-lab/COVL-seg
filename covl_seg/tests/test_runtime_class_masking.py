from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load_runtime_class_masking_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "covl_seg_d2_runtime"
        / "cat_seg"
        / "utils"
        / "class_masking.py"
    )
    spec = importlib.util.spec_from_file_location("runtime_class_masking", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mask_logits_and_targets_to_visible_classes_selects_only_visible_channels():
    module = _load_runtime_class_masking_module()

    logits = torch.arange(2 * 2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 2, 4)
    targets = torch.arange(2 * 2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 2, 4) * 10.0
    visible = torch.tensor([1, 3], dtype=torch.long)

    masked_logits, masked_targets = module.mask_logits_and_targets_to_visible_classes(
        logits=logits,
        targets=targets,
        visible_class_indexes=visible,
    )

    assert masked_logits.shape == (2, 2, 2, 2)
    assert masked_targets.shape == (2, 2, 2, 2)
    assert torch.equal(masked_logits, logits[..., [1, 3]])
    assert torch.equal(masked_targets, targets[..., [1, 3]])


def test_mask_logits_and_targets_to_visible_classes_returns_inputs_when_no_visible_indexes():
    module = _load_runtime_class_masking_module()

    logits = torch.randn(1, 2, 2, 3)
    targets = torch.randn(1, 2, 2, 3)

    out_logits, out_targets = module.mask_logits_and_targets_to_visible_classes(
        logits=logits,
        targets=targets,
        visible_class_indexes=None,
    )

    assert out_logits is logits
    assert out_targets is targets
