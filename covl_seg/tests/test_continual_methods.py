import pytest

from covl_seg.continual.methods import build_continual_method


def test_method_factory_returns_expected_type():
    method = build_continual_method("replay", config={})
    assert method.name == "replay"


def test_method_factory_rejects_unknown_method():
    with pytest.raises(ValueError):
        build_continual_method("unknown", config={})
