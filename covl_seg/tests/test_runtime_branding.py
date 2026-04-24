from pathlib import Path


def test_vendor_train_entry_uses_covl_seg_branding_imports():
    root = Path(__file__).resolve().parents[2]
    train_entry = root / "covl_seg" / "vendor" / "covl_seg_d2_runtime" / "train_net.py"
    text = train_entry.read_text(encoding="utf-8")
    assert "from covl_seg_d2 import (" in text


def test_vendor_cat_seg_runtime_exports_catseg_meta_architecture_name():
    root = Path(__file__).resolve().parents[2]
    runtime_root = root / "covl_seg" / "vendor" / "covl_seg_d2_runtime" / "cat_seg"
    model_text = (runtime_root / "cat_seg_model.py").read_text(encoding="utf-8")
    init_text = (runtime_root / "__init__.py").read_text(encoding="utf-8")

    assert "class CATSeg(" in model_text or "class COVLSeg(" in model_text
    assert "CATSeg" in init_text


def test_vendor_cat_seg_runtime_exports_covlseg_symbol_for_bridge_imports():
    root = Path(__file__).resolve().parents[2]
    runtime_root = root / "covl_seg" / "vendor" / "covl_seg_d2_runtime"
    cat_seg_init_text = (runtime_root / "cat_seg" / "__init__.py").read_text(encoding="utf-8")
    bridge_init_text = (runtime_root / "covl_seg_d2" / "__init__.py").read_text(encoding="utf-8")

    assert "from cat_seg import COVLSeg" in bridge_init_text
    assert "COVLSeg =" in cat_seg_init_text or "from .cat_seg_model import COVLSeg" in cat_seg_init_text
