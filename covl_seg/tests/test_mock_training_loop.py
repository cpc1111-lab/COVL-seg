import copy

import pytest
import torch

from covl_seg.continual.task_partition import TaskDef
from covl_seg.engine.detectron2_runner import build_covl_training_model
from covl_seg.engine.mock_training_loop import run_mock_task_training


def _state_dict_delta(before, after) -> float:
    delta = 0.0
    for key, tensor in before.items():
        if key not in after:
            continue
        if not torch.is_tensor(tensor) or not torch.is_tensor(after[key]):
            continue
        delta += float(torch.sum(torch.abs(after[key] - tensor)).item())
    return delta


def test_run_mock_task_training_updates_model_parameters():
    model = build_covl_training_model(num_classes=8, text_dim=16, seed=11)
    task = TaskDef(task_id=1, new_classes=[0, 1], seen_classes=[0, 1], background_classes=[2, 3, 4, 5, 6, 7])
    cfg = {
        "n_pre": 3,
        "n_main": 4,
        "enable_ciba": True,
        "enable_ctr": True,
        "enable_spectral_ogp": False,
        "batch_size": 2,
        "image_size": 16,
        "lr": 1e-3,
        "seed": 11,
    }

    before = copy.deepcopy(model.state_dict())
    out_model, phase_metrics = run_mock_task_training(model=model, task=task, cfg=cfg, basis_history=[])
    delta = _state_dict_delta(before, out_model.state_dict())

    assert delta > 0.0
    assert isinstance(phase_metrics, dict)
    assert set(phase_metrics.keys()) == {"phase1", "phase2", "phase3", "phase4"}


def test_phase1_loss_can_be_negative_when_ciba_enabled():
    model = build_covl_training_model(num_classes=6, text_dim=16, seed=3)
    task = TaskDef(task_id=2, new_classes=[1, 2], seen_classes=[1, 2], background_classes=[0, 3, 4, 5])
    cfg = {
        "n_pre": 2,
        "n_main": 1,
        "enable_ciba": True,
        "enable_ctr": False,
        "enable_spectral_ogp": False,
        "batch_size": 2,
        "image_size": 16,
        "seed": 3,
    }

    _, phase_metrics = run_mock_task_training(model=model, task=task, cfg=cfg, basis_history=[])
    phase1 = phase_metrics["phase1"]

    assert phase1["loss"] < 0.0
    assert phase1["ciba_loss"] < 0.0
    assert "beta_2_star" in phase1


def test_phase2_projection_reduces_norm_for_matching_basis():
    model = build_covl_training_model(num_classes=7, text_dim=16, seed=21)
    task = TaskDef(task_id=3, new_classes=[0, 1], seen_classes=[0, 1, 2], background_classes=[3, 4, 5, 6])
    basis_history = [
        [0.0 for _ in range(sum(p.numel() for p in model.core_model.parameters() if p.requires_grad))],
    ]
    basis_history[0][0] = 1.0
    cfg = {
        "n_pre": 1,
        "n_main": 3,
        "enable_ciba": False,
        "enable_ctr": True,
        "enable_spectral_ogp": True,
        "batch_size": 2,
        "image_size": 16,
        "seed": 21,
    }

    _, phase_metrics = run_mock_task_training(model=model, task=task, cfg=cfg, basis_history=basis_history)
    phase2 = phase_metrics["phase2"]

    assert phase2["proj_norm_before"] > 0.0
    assert phase2["proj_norm_after"] < phase2["proj_norm_before"]
    assert "oldfix_weighted_term" in phase2


def test_run_mock_task_training_cuda_completes_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    model = build_covl_training_model(num_classes=6, text_dim=16, seed=7).to("cuda")
    task = TaskDef(task_id=4, new_classes=[0, 1], seen_classes=[0, 1], background_classes=[2, 3, 4, 5])
    cfg = {
        "n_pre": 1,
        "n_main": 1,
        "enable_ciba": True,
        "enable_ctr": True,
        "enable_spectral_ogp": True,
        "batch_size": 1,
        "image_size": 8,
        "seed": 7,
    }

    _, phase_metrics = run_mock_task_training(model=model, task=task, cfg=cfg, basis_history=[])

    assert set(phase_metrics.keys()) == {"phase1", "phase2", "phase3", "phase4"}
