import pytest

open_clip = pytest.importorskip("open_clip")
import torch


def _make_visual_encoder(**overrides):
    from covl_seg.model.clip_encoder import CLIPVisualEncoder

    defaults = dict(model_name="ViT-B-16", pretrained="openai", clip_finetune="none")
    defaults.update(overrides)
    return CLIPVisualEncoder(**defaults)


def _make_text_encoder(**overrides):
    from covl_seg.model.clip_encoder import CLIPTextEncoder

    defaults = dict(model_name="ViT-B-16", pretrained="openai")
    defaults.update(overrides)
    return CLIPTextEncoder(**defaults)


def test_clip_encoder_produces_dense_features():
    encoder = _make_visual_encoder()
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = encoder(images)
    assert out.shape[0] == 1
    assert out.shape[-1] == 768
    assert out.ndim == 3


def test_clip_encoder_registers_forward_hooks():
    encoder = _make_visual_encoder(hook_layers=[3, 7])
    images = torch.randn(1, 3, 224, 224)
    encoder.intermediate_features.clear()
    with torch.no_grad():
        encoder(images)
    assert len(encoder.intermediate_features) == 2
    for feat in encoder.intermediate_features:
        assert feat.ndim == 3
        assert feat.shape[-1] == 768


def test_clip_finetune_attention_requires_grad():
    encoder = _make_visual_encoder(clip_finetune="attention")
    attn_trainable = []
    for name, param in encoder.clip_model.visual.named_parameters():
        if "attn" in name:
            attn_trainable.append(param.requires_grad)
        else:
            assert not param.requires_grad, f"{name} should be frozen in attention mode"
    assert any(attn_trainable), "At least one attn parameter should be trainable"


def test_clip_finetune_none_freezes_all():
    encoder = _make_visual_encoder(clip_finetune="none")
    for param in encoder.clip_model.parameters():
        assert not param.requires_grad


def test_text_encoder_produces_embeddings():
    encoder = _make_text_encoder()
    class_names = ["dog", "cat", "bird"]
    out = encoder(class_names)
    assert out.shape == (3, 768)
    assert out.requires_grad