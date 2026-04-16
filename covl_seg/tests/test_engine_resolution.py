import pytest


def test_resolve_train_engine_auto_without_detectron2_falls_back_to_mock():
    from covl_seg.scripts.train_continual import resolve_engine

    assert resolve_engine(requested="auto", detectron2_ready=False) == "mock"


def test_resolve_train_engine_d2_without_detectron2_raises():
    from covl_seg.scripts.train_continual import resolve_engine

    with pytest.raises(RuntimeError, match="Detectron2"):
        resolve_engine(requested="d2", detectron2_ready=False)


def test_resolve_eval_engine_auto_without_detectron2_falls_back_to_mock():
    from covl_seg.scripts.eval_continual import resolve_engine

    assert resolve_engine(requested="auto", detectron2_ready=False) == "mock"
