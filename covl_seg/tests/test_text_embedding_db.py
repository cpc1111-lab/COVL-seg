import pytest
import torch
from unittest.mock import MagicMock, patch


def _mock_text_encoder(output_dim=768):
    mock_encoder = MagicMock()
    mock_encoder.output_dim = output_dim

    def _forward(class_names):
        k = len(class_names)
        return torch.randn(k, output_dim, requires_grad=True)

    mock_encoder.side_effect = _forward
    mock_encoder.return_value = _forward([])
    return mock_encoder


@patch("covl_seg.data.text_embedding_db.CLIPTextEncoder")
def test_text_embedding_db_encodes(MockCLIPTextEncoder):
    mock_encoder = _mock_text_encoder(output_dim=512)
    MockCLIPTextEncoder.return_value = mock_encoder

    from covl_seg.data.text_embedding_db import TextEmbeddingDB

    db = TextEmbeddingDB(clip_model_name="ViT-B-16")
    class_names = ["dog", "cat", "bird"]
    result = db.encode(class_names)

    assert result.shape == (3, 512)
    assert result.requires_grad
    mock_encoder.assert_called_once_with(class_names)


@patch("covl_seg.data.text_embedding_db.CLIPTextEncoder")
def test_text_embedding_db_caches(MockCLIPTextEncoder):
    mock_encoder = _mock_text_encoder(output_dim=256)
    MockCLIPTextEncoder.return_value = mock_encoder

    from covl_seg.data.text_embedding_db import TextEmbeddingDB

    db = TextEmbeddingDB(clip_model_name="ViT-B-16")
    class_names = ["dog", "cat"]

    result1 = db.encode_with_cache(class_names)
    assert result1.shape == (2, 256)
    assert mock_encoder.call_count == 1

    result2 = db.encode_with_cache(class_names)
    assert result2.shape == (2, 256)
    assert mock_encoder.call_count == 1, "encoder should not be called again on cache hit"


@patch("covl_seg.data.text_embedding_db.CLIPTextEncoder")
def test_text_embedding_db_clear_cache(MockCLIPTextEncoder):
    mock_encoder = _mock_text_encoder(output_dim=128)
    MockCLIPTextEncoder.return_value = mock_encoder

    from covl_seg.data.text_embedding_db import TextEmbeddingDB

    db = TextEmbeddingDB(clip_model_name="ViT-B-16")
    class_names = ["aeroplane", "bicycle"]

    db.encode_with_cache(class_names)
    assert mock_encoder.call_count == 1

    db.clear_cache()

    db.encode_with_cache(class_names)
    assert mock_encoder.call_count == 2, "encoder should be called again after cache clear"