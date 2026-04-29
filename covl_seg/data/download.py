"""Automatic dataset download and preparation for COVL-Seg."""

import logging
import zipfile
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)

_COCO_STUFF_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "train_annotations": "http://images.cocodataset.org/annotations/stuff_ann_trainval_2017.zip",
}

_ADE20K_URLS = {
    "archive": "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
}


def _download_file(url: str, dest: Path, desc: str = "") -> Path:
    import urllib.request
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        _log.info("%s already exists: %s", desc, dest)
        return dest

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    _log.info("Downloading %s from %s ...", desc, url)
    try:
        with urllib.request.urlopen(url) as resp:
            with open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f)
        tmp.rename(dest)
        _log.info("Downloaded %s -> %s", desc, dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def _extract_zip(zip_path: Path, dest_dir: Path, strip_prefix: Optional[str] = None) -> None:
    _log.info("Extracting %s -> %s ...", zip_path, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename
            if strip_prefix and name.startswith(strip_prefix):
                name = name[len(strip_prefix):]
            if not name:
                continue
            target = dest_dir / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    _log.info("Extracted to %s", dest_dir)


def ensure_coco_stuff(root: str = "datasets/coco-stuff") -> Path:
    """Download and prepare COCO-Stuff-164 dataset if not found.

    Expected final structure:
        root/
            images/train2017/*.jpg
            images/val2017/*.jpg
            annotations/train2017/*.png
            annotations/val2017/*.png
    """
    root_path = Path(root).resolve()
    images_train = root_path / "images" / "train2017"
    annots_train = root_path / "annotations" / "train2017"

    if images_train.exists() and any(images_train.glob("*.jpg")) and annots_train.exists() and any(annots_train.glob("*.png")):
        _log.info("COCO-Stuff dataset found at %s", root_path)
        return root_path

    _log.info("COCO-Stuff dataset not found at %s, starting download...", root_path)
    archive_dir = root_path / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    train_zip = _download_file(_COCO_STUFF_URLS["train_images"], archive_dir / "train2017.zip", "COCO train2017 images")
    _extract_zip(train_zip, root_path / "images", strip_prefix=None)

    val_zip = _download_file(_COCO_STUFF_URLS["val_images"], archive_dir / "val2017.zip", "COCO val2017 images")
    _extract_zip(val_zip, root_path / "images", strip_prefix=None)

    ann_zip = _download_file(_COCO_STUFF_URLS["train_annotations"], archive_dir / "stuff_ann_trainval_2017.zip", "COCO-Stuff annotations")
    _extract_zip(ann_zip, root_path / "annotations", strip_prefix=None)

    _renamed_coco_annotations(root_path)
    _log.info("COCO-Stuff dataset ready at %s", root_path)
    return root_path


def _renamed_coco_annotations(root_path: Path) -> None:
    """Convert instance annotation PNGs to stuff annotation PNGs if needed.

    COCO-Stuff annotations are inside annotations/stuff_{train,val}2017/
    We move them to annotations/train2017/ and annotations/val2017/.
    Also handles the case where annotations are already in the right place.
    """
    for split_dir_name, alt_name in [("train2017", "stuff_train2017"), ("val2017", "stuff_val2017")]:
        target_dir = root_path / "annotations" / split_dir_name
        alt_dir = root_path / "annotations" / alt_name
        if target_dir.exists() and any(target_dir.glob("*.png")):
            continue
        if alt_dir.exists():
            _log.info("Moving %s -> %s", alt_dir, target_dir)
            if target_dir.exists():
                for f in alt_dir.glob("*.png"):
                    f.rename(target_dir / f.name)
            else:
                alt_dir.rename(target_dir)
        json_dir = root_path / "annotations"
        json_sub = json_dir / alt_name
        if json_sub.is_dir() and not target_dir.exists():
            json_sub.rename(target_dir)


def ensure_ade20k(root: str = "datasets/ADE20K") -> Path:
    """Download and prepare ADE20K dataset if not found.

    Expected final structure:
        root/
            images/training/*.jpg
            images/validation/*.jpg
            annotations/training/*.png
            annotations/validation/*.png
    """
    root_path = Path(root).resolve()
    images_train = root_path / "images" / "training"
    annots_train = root_path / "annotations" / "training"

    if images_train.exists() and any(images_train.glob("*.jpg")) and annots_train.exists() and any(annots_train.glob("*.png")):
        _log.info("ADE20K dataset found at %s", root_path)
        return root_path

    _log.info("ADE20K dataset not found at %s, starting download...", root_path)
    archive_dir = root_path / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    zip_path = _download_file(_ADE20K_URLS["archive"], archive_dir / "ADEChallengeData2016.zip", "ADE20K")

    extract_dir = root_path / "ADEChallengeData2016"
    _extract_zip(zip_path, extract_dir)

    ade_inner = extract_dir / "ADEChallengeData2016"
    if not ade_inner.exists():
        ade_inner = extract_dir

    for subdir in ["images/training", "images/validation", "annotations/training", "annotations/validation"]:
        src = ade_inner / subdir
        dst = root_path / subdir
        if src.exists() and not dst.exists():
            src.rename(dst)

    if images_train.exists() and any(images_train.glob("*.jpg")):
        _log.info("ADE20K dataset ready at %s", root_path)
    else:
        _log.warning("ADE20K extraction may need manual reorganization. Check %s", root_path)

    return root_path


def ensure_dataset(config_path: str, dataset_root: Optional[str] = None) -> str:
    """Auto-detect dataset type from config and download if needed."""
    cfg_lower = config_path.lower()
    if dataset_root is None:
        if "coco" in cfg_lower:
            dataset_root = "datasets/coco-stuff"
        else:
            dataset_root = "datasets/ADE20K"

    if "coco" in cfg_lower:
        ensure_coco_stuff(dataset_root)
    else:
        ensure_ade20k(dataset_root)

    return dataset_root


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(description="Download COVL-Seg datasets")
    p.add_argument("--dataset", choices=["coco-stuff", "ade20k"], required=True)
    p.add_argument("--root", default=None)
    args = p.parse_args()

    if args.dataset == "coco-stuff":
        root = args.root or "datasets/coco-stuff"
        ensure_coco_stuff(root)
    elif args.dataset == "ade20k":
        root = args.root or "datasets/ADE20K"
        ensure_ade20k(root)