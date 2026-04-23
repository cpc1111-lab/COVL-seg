from pathlib import Path


def test_vendor_train_entry_uses_covl_seg_branding_imports():
    root = Path(__file__).resolve().parents[2]
    train_entry = root / "covl_seg" / "vendor" / "covl_seg_d2_runtime" / "train_net.py"
    text = train_entry.read_text(encoding="utf-8")
    assert "from covl_seg_d2 import (" in text


def test_vendor_runtime_config_defines_covl_toggle_keys():
    root = Path(__file__).resolve().parents[2]
    config_file = root / "covl_seg" / "vendor" / "covl_seg_d2_runtime" / "cat_seg" / "config.py"
    text = config_file.read_text(encoding="utf-8")

    assert "cfg.MODEL.COVL.ENABLE_CIBA" in text
    assert "cfg.MODEL.COVL.ENABLE_CTR" in text
    assert "cfg.MODEL.COVL.ENABLE_OGP" in text
