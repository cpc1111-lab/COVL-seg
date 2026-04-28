import torch
import pytest


def _make_head(num_classes=5, feature_dim=32):
    from covl_seg.model.fusion_head import ContinualFusionHead

    return ContinualFusionHead(num_classes=num_classes, feature_dim=feature_dim)


def test_fusion_head_basic():
    head = _make_head(num_classes=5, feature_dim=32)
    B, K, H, W = 2, 5, 16, 16
    backbone_logits = torch.randn(B, K, H, W)
    clip_logits = torch.randn(B, K, H, W)
    text_embeddings = torch.randn(B, K, 32)

    out = head(backbone_logits, clip_logits, text_embeddings)
    assert out.shape == (B, K, H, W)


def test_fusion_head_alpha_tau_injectable():
    head = _make_head(num_classes=3, feature_dim=16)
    B, K, H, W = 1, 3, 8, 8
    backbone_logits = torch.randn(B, K, H, W)
    clip_logits = torch.randn(B, K, H, W)
    text_embeddings = torch.randn(B, K, 16)

    out1 = head(backbone_logits, clip_logits, text_embeddings)

    head.alpha.data.fill_(0.9)
    head.tau.data.fill_(2.0)

    out2 = head(backbone_logits, clip_logits, text_embeddings)
    assert out2.shape == (B, K, H, W)
    assert not torch.allclose(out1, out2)


def test_fusion_head_alpha_tau_gradients():
    head = _make_head(num_classes=4, feature_dim=16)
    B, K, H, W = 2, 4, 12, 12
    backbone_logits = torch.randn(B, K, H, W)
    clip_logits = torch.randn(B, K, H, W)
    text_embeddings = torch.randn(B, K, 16)

    out = head(backbone_logits, clip_logits, text_embeddings)
    loss = out.sum()
    loss.backward()

    assert head.alpha.grad is not None
    assert head.tau.grad is not None
    assert not torch.all(head.alpha.grad == 0)
    assert not torch.all(head.tau.grad == 0)


def test_fusion_head_different_sizes():
    for K, H, W, D in [(3, 8, 8, 16), (10, 32, 64, 64), (1, 4, 4, 8)]:
        head = _make_head(num_classes=K, feature_dim=D)
        backbone_logits = torch.randn(1, K, H, W)
        clip_logits = torch.randn(1, K, H, W)
        text_embeddings = torch.randn(1, K, D)

        out = head(backbone_logits, clip_logits, text_embeddings)
        assert out.shape == (1, K, H, W)