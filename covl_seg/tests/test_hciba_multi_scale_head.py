import torch
import pytest

from covl_seg.model.hciba_multi_scale_head import HCIBAMultiScaleHead


def _make_head(clip_dim=768, dino_dims=(384, 384, 768), out_dim=768, num_heads=8):
    return HCIBAMultiScaleHead(
        clip_dim=clip_dim, dino_dims=dino_dims, out_dim=out_dim, num_heads=num_heads
    )


def _make_inputs(B=2, H=56, W=56, clip_dim=768, dino_dims=(384, 384, 768)):
    clip_features = torch.randn(B, 10, clip_dim)
    strides = {"res3": 8, "res4": 16, "res5": 32}
    dino_features = {}
    for key, dim in zip(["res3", "res4", "res5"], dino_dims):
        s = strides[key]
        dino_features[key] = torch.randn(B, dim, H // s, W // s)
    return clip_features, dino_features


def test_multi_scale_head_output_shape():
    head = _make_head()
    clip_features, dino_features = _make_inputs()
    output = head(clip_features, dino_features)
    B, _, H3, W3 = dino_features["res3"].shape
    assert output.shape == (B, 768, H3, W3)


def test_multi_scale_head_learnable():
    head = _make_head()
    trainable = [p for p in head.parameters() if p.requires_grad]
    assert len(trainable) > 0
    total_params = sum(p.numel() for p in head.parameters())
    assert total_params > 0


def test_multi_scale_head_different_resolutions():
    head = _make_head()
    for H, W in [(56, 56), (112, 84), (64, 32)]:
        clip_features, dino_features = _make_inputs(H=H, W=W)
        output = head(clip_features, dino_features)
        B = clip_features.shape[0]
        assert output.shape[2] == H // 8
        assert output.shape[3] == W // 8


def test_multi_scale_head_custom_dims():
    clip_dim = 512
    dino_dims = (256, 256, 512)
    out_dim = 256
    head = _make_head(clip_dim=clip_dim, dino_dims=dino_dims, out_dim=out_dim)
    clip_features, dino_features = _make_inputs(
        clip_dim=clip_dim, dino_dims=dino_dims
    )
    output = head(clip_features, dino_features)
    B, _, H3, W3 = dino_features["res3"].shape
    assert output.shape == (B, out_dim, H3, W3)


def test_multi_scale_head_backward_pass():
    head = _make_head()
    clip_features, dino_features = _make_inputs()
    clip_features.requires_grad_(True)
    output = head(clip_features, dino_features)
    loss = output.sum()
    loss.backward()
    assert clip_features.grad is not None
    assert torch.isfinite(clip_features.grad).all()