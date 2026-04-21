import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile


OPEN_VOCAB_SOURCES = {
    "voc2012": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "filename": "VOCtrainval_11-May-2012.tar",
        "kind": "tar",
    },
    "voc2012_aug": {
        "url": "https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=1",
        "filename": "SegmentationClassAug.zip",
        "kind": "zip",
    },
    "voc2010": {
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar",
        "filename": "VOCtrainval_03-May-2010.tar",
        "kind": "tar",
    },
    "pc459_trainval": {
        "url": "https://roozbehm.info/pascal-context/trainval.tar.gz",
        "filename": "trainval.tar.gz",
        "kind": "tar",
    },
}


def _stream_copy_with_progress(
    source: BinaryIO,
    target: BinaryIO,
    total_bytes: Optional[int],
    label: str,
    phase: str,
    chunk_size: int = 1024 * 1024,
) -> None:
    copied = 0
    last_pct = -1
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            break
        target.write(chunk)
        copied += len(chunk)
        if total_bytes is None:
            print(f"{phase.capitalize()} progress [{label}]: {copied / (1024 * 1024):.1f} MB")
            continue
        pct = int((copied * 100) / max(total_bytes, 1))
        if pct >= last_pct + 5 or pct == 100:
            last_pct = pct
            print(f"{phase.capitalize()} progress [{label}]: {pct}% ({copied}/{total_bytes} bytes)")


def _read_content_length(response: BinaryIO) -> Optional[int]:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("Content-Length")
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _is_valid_zip(path: Path) -> bool:
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None
    except (BadZipFile, OSError):
        return False


def _is_valid_tar(path: Path) -> bool:
    try:
        with tarfile.open(path) as archive:
            return archive.getmembers() is not None
    except (tarfile.TarError, OSError):
        return False


def _is_valid_artifact(path: Path, kind: str) -> bool:
    if kind == "zip":
        return _is_valid_zip(path)
    if kind == "tar":
        return _is_valid_tar(path)
    return path.is_file() and path.stat().st_size > 0


def download_file(url: str, dest: Path, kind: str, force_download: bool = False) -> Path:
    if dest.exists() and not force_download:
        if _is_valid_artifact(dest, kind=kind):
            print(f"Skipping download, file already exists: {dest}")
            return dest
        print(f"Existing file is invalid and will be re-downloaded: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    if tmp.exists():
        tmp.unlink()

    print(f"Downloading {url} -> {dest}")
    with urlopen(url, timeout=120) as response, tmp.open("wb") as out_file:
        total = _read_content_length(response)
        _stream_copy_with_progress(response, out_file, total, dest.name, "download")
    if not _is_valid_artifact(tmp, kind=kind):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded artifact is invalid for kind={kind}: {dest.name}")
    tmp.replace(dest)
    return dest


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        members = archive.infolist()
        total = len(members)
        last_pct = -1
        for idx, member in enumerate(members, start=1):
            archive.extract(member, target_dir)
            pct = int((idx * 100) / max(total, 1))
            if pct >= last_pct + 10 or pct == 100:
                last_pct = pct
                print(f"Extract progress [{zip_path.name}]: {pct}% ({idx}/{total} entries)")


def _safe_extract_tar(tar_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as archive:
        members = archive.getmembers()
        total = len(members)
        last_pct = -1
        for idx, member in enumerate(members, start=1):
            archive.extract(member, target_dir)
            pct = int((idx * 100) / max(total, 1))
            if pct >= last_pct + 10 or pct == 100:
                last_pct = pct
                print(f"Extract progress [{tar_path.name}]: {pct}% ({idx}/{total} entries)")


def resolve_datasets_root(cli_value: Optional[str], repo_root: Path) -> Path:
    if cli_value:
        return Path(cli_value)
    env_value = os.environ.get("DETECTRON2_DATASETS")
    if env_value:
        return Path(env_value)
    return repo_root / "datasets"


def _voc_outputs_ready(datasets_root: Path) -> bool:
    voc2012 = datasets_root / "VOCdevkit" / "VOC2012"
    return (
        (voc2012 / "annotations_detectron2" / "val").is_dir()
        and (voc2012 / "annotations_detectron2_bg" / "val").is_dir()
    )


def _pc59_ready(datasets_root: Path) -> bool:
    return (datasets_root / "VOCdevkit" / "VOC2010" / "annotations_detectron2" / "pc59_val").is_dir()


def _pc459_ready(datasets_root: Path) -> bool:
    return (datasets_root / "VOCdevkit" / "VOC2010" / "annotations_detectron2" / "pc459_val").is_dir()


def _run_prepare_script(script_name: str, runtime_root: Path, datasets_root: Path) -> None:
    script = runtime_root / "datasets" / script_name
    env = os.environ.copy()
    env["DETECTRON2_DATASETS"] = str(datasets_root)
    subprocess.run(["python", str(script)], check=True, env=env)


def _ensure_pascal_context_metadata(datasets_root: Path) -> None:
    voc2010 = datasets_root / "VOCdevkit" / "VOC2010"
    val_split = voc2010 / "pascalcontext_val.txt"
    if val_split.exists():
        return

    fallback = voc2010 / "ImageSets" / "Segmentation" / "val.txt"
    if fallback.exists():
        print(f"Creating missing pascalcontext_val.txt from {fallback}")
        val_split.write_text(fallback.read_text(encoding="utf-8"), encoding="utf-8")
        return

    raise FileNotFoundError(
        "Missing Pascal Context validation split. Expected one of: "
        f"{val_split} or {fallback}"
    )


def ensure_open_vocab_eval_data_ready(
    datasets_root: Path,
    runtime_root: Path,
    force_download: bool = False,
) -> None:
    if _voc_outputs_ready(datasets_root) and _pc59_ready(datasets_root) and _pc459_ready(datasets_root) and not force_download:
        print("Found prepared VOC/PC59/PC459 outputs; skipping download and conversion")
        return

    archive_root = datasets_root / "open_vocab_archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    voc_root = datasets_root / "VOCdevkit"
    voc_root.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Path] = {}
    for spec in OPEN_VOCAB_SOURCES.values():
        path = archive_root / spec["filename"]
        artifacts[spec["filename"]] = download_file(
            spec["url"],
            path,
            kind=spec["kind"],
            force_download=force_download,
        )

    _safe_extract_tar(artifacts["VOCtrainval_11-May-2012.tar"], datasets_root)
    _safe_extract_tar(artifacts["VOCtrainval_03-May-2010.tar"], datasets_root)
    _safe_extract_tar(artifacts["trainval.tar.gz"], datasets_root / "VOCdevkit" / "VOC2010")

    with tempfile.TemporaryDirectory(prefix="voc_aug_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        _safe_extract_zip(artifacts["SegmentationClassAug.zip"], tmp_path)
        aug_dir = tmp_path / "SegmentationClassAug"
        target_aug = datasets_root / "VOCdevkit" / "VOC2012" / "SegmentationClassAug"
        if target_aug.exists():
            shutil.rmtree(target_aug)
        shutil.copytree(aug_dir, target_aug)

    _ensure_pascal_context_metadata(datasets_root)

    _run_prepare_script("prepare_voc.py", runtime_root=runtime_root, datasets_root=datasets_root)
    _run_prepare_script("prepare_pascal_context_59.py", runtime_root=runtime_root, datasets_root=datasets_root)
    _run_prepare_script("prepare_pascal_context_459.py", runtime_root=runtime_root, datasets_root=datasets_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare open-vocab eval datasets")
    parser.add_argument("--datasets-root", default=None, help="Dataset root override")
    parser.add_argument(
        "--runtime-root",
        default="covl_seg/vendor/covl_seg_d2_runtime",
        help="Runtime root containing datasets prepare scripts",
    )
    parser.add_argument("--force-download", action="store_true", help="Redownload all archives")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    datasets_root = resolve_datasets_root(args.datasets_root, repo_root=repo_root)
    runtime_root = Path(args.runtime_root)
    if not runtime_root.is_absolute():
        runtime_root = (repo_root / runtime_root).resolve()
    ensure_open_vocab_eval_data_ready(
        datasets_root=datasets_root,
        runtime_root=runtime_root,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()
