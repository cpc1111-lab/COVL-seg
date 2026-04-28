import pytest

torch = pytest.importorskip("torch")

import torch
from torch import nn

from covl_seg.model.dino_extractor import DINOv2FeatureExtractor


class _MockBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class _MockViT(nn.Module):
    def __init__(self, num_blocks: int = 12, embed_dim: int = 768, patch_size: int = 14):
        super().__init__()
        self.blocks = nn.Sequential(*[_MockBlock(embed_dim) for _ in range(num_blocks)])
        self._patch_size = patch_size
        self._embed_dim = embed_dim
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        ps = self._patch_size
        x = x.reshape(B, C, H // ps, ps, W // ps, ps)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, (H // ps) * (W // ps), -1)
        x = self.proj(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.blocks(x)
        return x


def _make_extractor(num_blocks=12, embed_dim=768, model_name="dinov2_vitb14"):
    mock_model = _MockViT(num_blocks=num_blocks, embed_dim=embed_dim)
    return DINOv2FeatureExtractor(model_name=model_name, model=mock_model)


def test_dino_extractor_outputs_multi_scale():
    extractor = _make_extractor()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = extractor(images)
    assert set(out.keys()) == {"res3", "res4", "res5"}
    for key in ["res3", "res4", "res5"]:
        assert out[key].shape[0] == 2


def test_dino_extractor_stride_consistency():
    H, W = 224, 224
    extractor = _make_extractor()
    images = torch.randn(1, 3, H, W)
    with torch.no_grad():
        out = extractor(images)
    assert out["res3"].shape[2] == H // 8
    assert out["res3"].shape[3] == W // 8
    assert out["res4"].shape[2] == H // 16
    assert out["res4"].shape[3] == W // 16
    assert out["res5"].shape[2] == H // 32
    assert out["res5"].shape[3] == W // 32


def test_dino_extractor_freeze():
    extractor = _make_extractor()
    for param in extractor.model.parameters():
        assert not param.requires_grad


def test_dino_extractor_small_variant():
    extractor = _make_extractor(num_blocks=12, embed_dim=384, model_name="dinov2_vits14")
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = extractor(images)
    assert out["res3"].shape[1] == 384
    assert out["res4"].shape[1] == 384
    assert out["res5"].shape[1] == 384


def test_dino_extractor_large_variant():
    extractor = _make_extractor(num_blocks=24, embed_dim=1024, model_name="dinov2_vitl14")
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = extractor(images)
    assert out["res3"].shape[1] == 1024


def test_dino_extractor_no_freeze():
    mock_model = _MockViT()
    extractor = DINOv2FeatureExtractor(model_name="dinov2_vitb14", model=mock_model, freeze=False)
    assert any(p.requires_grad for p in extractor.model.parameters())