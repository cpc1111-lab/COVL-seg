import argparse
import os
import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO, List, Optional, Union
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile


COCO_ARCHIVES = [
    {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "name": "train2017.zip",
        "extract_subdir": Path("coco-stuff") / "images",
    },
    {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "name": "val2017.zip",
        "extract_subdir": Path("coco-stuff") / "images",
    },
    {
        "url": "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip",
        "name": "stuffthingmaps_trainval2017.zip",
        "extract_subdir": Path("coco-stuff") / "annotations",
    },
]


def download_file_with_retries(
    url: str,
    dest: Path,
    force_download: bool = False,
    retries: int = 3,
    timeout: int = 60,
) -> Path:
    if dest.exists() and not force_download:
        if _is_valid_zip(dest):
            print(f"Skipping download, valid archive already exists: {dest}")
            return dest
        print(f"Existing archive is invalid and will be re-downloaded: {dest}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_name(f"{dest.name}.part")
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        with suppress(FileNotFoundError):
            temp_dest.unlink()
        try:
            print(f"Downloading {url} -> {dest} (attempt {attempt}/{retries})")
            with urlopen(url, timeout=timeout) as response, temp_dest.open("wb") as out_file:
                total_bytes = _read_content_length(response)
                _stream_copy_with_progress(
                    source=response,
                    target=out_file,
                    total_bytes=total_bytes,
                    label=dest.name,
                    phase="download",
                )
            if not _is_valid_zip(temp_dest):
                raise RuntimeError(f"Downloaded file is not a valid zip archive: {temp_dest}")
            temp_dest.replace(dest)
            return dest
        except Exception as exc:  # pragma: no cover - exercised via monkeypatched failures
            last_error = exc
            with suppress(FileNotFoundError):
                temp_dest.unlink()
            if attempt < retries:
                print(f"Download failed for {url}: {exc}; retrying...")

    raise RuntimeError(f"Failed to download {url} after {retries} attempts") from last_error


def _is_valid_zip(path: Path) -> bool:
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None
    except (BadZipFile, OSError):
        return False


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


def _stream_copy_with_progress(
    source: BinaryIO,
    target: BinaryIO,
    total_bytes: Optional[int],
    label: str,
    phase: str,
    chunk_size: int = 1024 * 1024,
) -> None:
    written = 0
    last_pct = -1
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            break
        target.write(chunk)
        written += len(chunk)
        if total_bytes is None:
            print(f"{phase.capitalize()} progress [{label}]: {written / (1024 * 1024):.1f} MB")
            continue
        pct = int((written * 100) / max(total_bytes, 1))
        if pct >= last_pct + 5 or pct == 100:
            last_pct = pct
            print(f"{phase.capitalize()} progress [{label}]: {pct}% ({written}/{total_bytes} bytes)")


def _is_path_within_directory(base_dir: Path, candidate: Path) -> bool:
    base_resolved = base_dir.resolve()
    candidate_resolved = candidate.resolve()
    return candidate_resolved == base_resolved or base_resolved in candidate_resolved.parents


def extract_archive(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {target_dir}")
    with ZipFile(zip_path) as archive:
        members = archive.infolist()
        total_members = len(members)
        completed = 0
        last_pct = -1
        for member in members:
            member_path = target_dir / member.filename
            if not _is_path_within_directory(target_dir, member_path):
                raise ValueError(f"Unsafe archive entry: {member.filename}")
            archive.extract(member, target_dir)
            completed += 1
            pct = int((completed * 100) / max(total_members, 1))
            if pct >= last_pct + 10 or pct == 100:
                last_pct = pct
                print(
                    "Extract progress "
                    f"[{zip_path.name}]: {pct}% ({completed}/{total_members} entries)"
                )


def _build_env_with_datasets(datasets_root: Path) -> dict:
    env = os.environ.copy()
    env["DETECTRON2_DATASETS"] = str(datasets_root)
    return env


def _resolve_d2_runtime_root(repo_root: Path) -> Path:
    override = os.environ.get("COVL_SEG_D2_PROJECT_ROOT", "").strip()
    if override:
        return Path(override)
    return repo_root / "covl_seg" / "vendor" / "covl_seg_d2_runtime"


def run_prepare_coco_stuff(runtime_root: Path, datasets_root: Path) -> None:
    prepare_script = runtime_root / "datasets" / "prepare_coco_stuff.py"
    cmd = [sys.executable, str(prepare_script)]
    subprocess.run(cmd, check=True, env=_build_env_with_datasets(datasets_root))


def run_training_command(cmd: List[str], datasets_root: Path) -> None:
    subprocess.run(cmd, check=True, env=_build_env_with_datasets(datasets_root))


def validate_coco_layout(coco_root: Path) -> None:
    required_dirs = [
        coco_root / "coco-stuff" / "images" / "train2017",
        coco_root / "coco-stuff" / "images" / "val2017",
        coco_root / "coco-stuff" / "annotations" / "train2017",
        coco_root / "coco-stuff" / "annotations" / "val2017",
        coco_root / "coco-stuff" / "annotations_detectron2" / "train2017",
        coco_root / "coco-stuff" / "annotations_detectron2" / "val2017",
    ]
    missing_paths = [path for path in required_dirs if not path.is_dir()]
    if missing_paths:
        missing_list = "\n".join(f"- {path}" for path in missing_paths)
        raise ValueError(f"Missing required COCO-Stuff paths:\n{missing_list}")


def _coco_stuff_tree_exists(datasets_root: Path) -> bool:
    required_dirs = [
        datasets_root / "coco-stuff" / "images" / "train2017",
        datasets_root / "coco-stuff" / "images" / "val2017",
        datasets_root / "coco-stuff" / "annotations" / "train2017",
        datasets_root / "coco-stuff" / "annotations" / "val2017",
    ]
    return all(path.is_dir() for path in required_dirs)


def _coco_stuff_prepared_tree_exists(datasets_root: Path) -> bool:
    required_dirs = [
        datasets_root / "coco-stuff" / "annotations_detectron2" / "train2017",
        datasets_root / "coco-stuff" / "annotations_detectron2" / "val2017",
    ]
    return all(path.is_dir() for path in required_dirs)


def ensure_coco_stuff_ready_for_training(
    datasets_root: Path,
    runtime_root: Path,
    force_download: bool = False,
) -> None:
    coco_stuff_root = datasets_root / "coco-stuff"
    archive_root = datasets_root / "coco_archives"

    if _coco_stuff_tree_exists(datasets_root) and _coco_stuff_prepared_tree_exists(datasets_root) and not force_download:
        print(f"Found prepared COCO-Stuff tree: {coco_stuff_root}; skipping download/extract/prepare")
        validate_coco_layout(datasets_root)
        return

    if _coco_stuff_tree_exists(datasets_root) and not force_download:
        print(f"Found existing COCO-Stuff tree: {coco_stuff_root}")
    else:
        print(f"COCO-Stuff dataset missing under {datasets_root}, starting auto download+extract...")
        for spec in COCO_ARCHIVES:
            archive_path = archive_root / spec["name"]
            extract_target = datasets_root / spec["extract_subdir"]
            downloaded_archive = download_file_with_retries(
                url=spec["url"],
                dest=archive_path,
                force_download=force_download,
            )
            extract_archive(downloaded_archive, extract_target)

    if not _coco_stuff_prepared_tree_exists(datasets_root) or force_download:
        run_prepare_coco_stuff(runtime_root=runtime_root, datasets_root=datasets_root)
    else:
        print("Found existing prepared annotations_detectron2 tree; skipping prepare step")

    validate_coco_layout(datasets_root)


def build_train_command(
    config_path: Union[str, Path],
    output_dir: Union[str, Path],
    seed: int,
    max_tasks: Optional[int],
) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "covl_seg.scripts.train_continual",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--engine",
        "d2",
    ]
    if max_tasks is not None:
        command.extend(["--max-tasks", str(max_tasks)])
    return command


def resolve_datasets_root(cli_value: Optional[str], repo_root: Path) -> Path:
    if cli_value:
        return Path(cli_value)

    env_value = os.environ.get("DETECTRON2_DATASETS")
    if env_value:
        return Path(env_value)

    return repo_root / "datasets"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap COCO-Stuff data and launch continual training"
    )
    parser.add_argument("--datasets-root", default=None, help="Dataset root override")
    parser.add_argument(
        "--config",
        default="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        help="Path to training config",
    )
    parser.add_argument("--output-dir", required=True, help="Training output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional task cap")
    download_mode = parser.add_mutually_exclusive_group()
    download_mode.add_argument(
        "--force-download", action="store_true", help="Redownload archives"
    )
    download_mode.add_argument(
        "--skip-download", action="store_true", help="Skip download phase"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions only")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    datasets_root = resolve_datasets_root(args.datasets_root, repo_root=repo_root)
    runtime_root = _resolve_d2_runtime_root(repo_root=repo_root)
    coco_stuff_root = datasets_root / "coco-stuff"
    archive_root = datasets_root / "coco_archives"

    train_cmd = build_train_command(
        config_path=args.config,
        output_dir=args.output_dir,
        seed=args.seed,
        max_tasks=args.max_tasks,
    )

    if args.dry_run:
        print(f"[dry-run] repository root: {repo_root}")
        print(f"[dry-run] datasets root: {datasets_root}")
        if args.skip_download:
            print("[dry-run] skipping download/extract phase (--skip-download)")
        else:
            for spec in COCO_ARCHIVES:
                archive_path = archive_root / spec["name"]
                extract_target = datasets_root / spec["extract_subdir"]
                print(
                    "[dry-run] download "
                    f"{spec['url']} -> {archive_path} (force_download={args.force_download})"
                )
                print(f"[dry-run] extract {archive_path} -> {extract_target}")
        print(
            "[dry-run] run prepare command: "
            f"{sys.executable} {(runtime_root / 'datasets' / 'prepare_coco_stuff.py')}"
        )
        print(f"[dry-run] validate COCO layout under {datasets_root}")
        print(f"[dry-run] launch training command: {' '.join(train_cmd)}")
        return

    if not args.skip_download:
        ensure_coco_stuff_ready_for_training(
            datasets_root=datasets_root,
            runtime_root=runtime_root,
            force_download=args.force_download,
        )
    else:
        run_prepare_coco_stuff(runtime_root=runtime_root, datasets_root=datasets_root)
        validate_coco_layout(datasets_root)
    run_training_command(cmd=train_cmd, datasets_root=datasets_root)


if __name__ == "__main__":
    main()
