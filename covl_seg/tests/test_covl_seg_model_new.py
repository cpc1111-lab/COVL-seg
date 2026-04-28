"""Tests for COVLSegModelV2 with heavily mocked submodules."""

import torch
from torch import nn
from unittest.mock import MagicMock

from covl_seg.model.covl_seg_model_new import COVLSegModelV2
from covl_seg.model.fusion_head import ContinualFusionHead

B, K, H_IMG, W_IMG = 2, 5, 224, 224
CLIP_DIM = 64
OUT_DIM = 32
DINO_DIMS = (384, 384, 768)
H_OUT, W_OUT = H_IMG // 8, W_IMG // 8


def _create_mock_model():
    model = COVLSegModelV2.__new__(COVLSegModelV2)
    nn.Module.__init__(model)

    model.num_classes = K
    model.out_dim = OUT_DIM
    model.clip_dim = CLIP_DIM

    model.clip_visual = MagicMock()
    model.clip_visual.dim = CLIP_DIM
    model.clip_visual.get_dense_features = MagicMock(
        return_value=torch.randn(B, 10, CLIP_DIM)
    )

    model.clip_text = MagicMock(return_value=torch.randn(K, OUT_DIM))

    model.dino = MagicMock(
        return_value={
            "res3": torch.randn(B, DINO_DIMS[0], H_OUT, W_OUT),
            "res4": torch.randn(B, DINO_DIMS[1], H_IMG // 16, W_IMG // 16),
            "res5": torch.randn(B, DINO_DIMS[2], H_IMG // 32, W_IMG // 32),
        }
    )

    model.hciba_head = MagicMock(
        return_value=torch.randn(B, OUT_DIM, H_OUT, W_OUT)
    )

    model.fusion_head = ContinualFusionHead(num_classes=K, feature_dim=OUT_DIM)
    model.clip_logit_proj = nn.Linear(CLIP_DIM, OUT_DIM, bias=False)

    return model


def test_model_forward_produces_logits():
    model = _create_mock_model()
    images = torch.randn(B, 3, H_IMG, W_IMG)
    class_names = ["dog", "cat", "bird", "car", "tree"]

    output = model(images, class_names)

    assert "logits" in output
    logits = output["logits"]
    assert logits.shape == (B, K, H_OUT, W_OUT)


def test_model_training_loss():
    model = _create_mock_model()
    images = torch.randn(B, 3, H_IMG, W_IMG)
    class_names = ["dog", "cat", "bird", "car", "tree"]
    targets = torch.randint(0, K, (B, H_OUT, W_OUT))

    output = model(images, class_names, targets=targets)

    assert "loss" in output
    assert "logits" in output
    loss = output["loss"]
    assert loss.ndim == 0
    assert loss.requires_grad


def test_model_inject_alpha_tau():
    model = _create_mock_model()

    model.inject_alpha_tau(alpha=0.9, tau=2.0)

    assert torch.allclose(model.fusion_head.alpha.data, torch.tensor(0.9))
    assert torch.allclose(model.fusion_head.tau.data, torch.tensor(2.0))


def test_model_forward_no_loss_without_targets():
    model = _create_mock_model()
    images = torch.randn(B, 3, H_IMG, W_IMG)
    class_names = ["dog", "cat", "bird", "car", "tree"]

    output = model(images, class_names)

    assert "logits" in output
    assert "loss" not in output