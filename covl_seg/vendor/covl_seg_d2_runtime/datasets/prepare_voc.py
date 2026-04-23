"""Prepare VOC metadata for COVL-Seg runtime.

This script intentionally avoids deprecated NumPy string aliases like ``np.str``.
"""

from __future__ import annotations

from pathlib import Path


def ensure_pascal_context_val_split(voc2010_root: Path) -> Path:
    seg_dir = voc2010_root / "ImageSets" / "Segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    src = seg_dir / "val.txt"
    dst = voc2010_root / "pascalcontext_val.txt"
    if src.exists() and not dst.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dst


if __name__ == "__main__":
    root = Path("datasets") / "VOCdevkit" / "VOC2010"
    out = ensure_pascal_context_val_split(root)
    print(f"Prepared: {out}")
