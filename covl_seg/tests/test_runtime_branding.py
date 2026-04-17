from pathlib import Path


def test_vendor_train_entry_uses_covl_seg_branding_imports():
    root = Path(__file__).resolve().parents[2]
    train_entry = root / "covl_seg" / "vendor" / "covl_seg_d2_runtime" / "train_net.py"
    text = train_entry.read_text(encoding="utf-8")
    assert "from covl_seg_d2 import (" in text
