import json
from pathlib import Path
from typing import Iterable, List


def _write_split(file_path: Path, class_groups: Iterable[List[int]]) -> Path:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tasks": [
            {"task_id": idx, "classes": group}
            for idx, group in enumerate(class_groups)
        ]
    }
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return file_path


def project_split_dir() -> Path:
    return Path(__file__).resolve().parent / "splits"


def contiguous_groups(total: int, group_sizes: List[int]) -> List[List[int]]:
    classes = list(range(total))
    out: List[List[int]] = []
    cursor = 0
    for size in group_sizes:
        out.append(classes[cursor : cursor + size])
        cursor += size
    if cursor != total:
        raise ValueError("group sizes must sum to total")
    return out


def ensure_split(file_name: str, class_groups: List[List[int]]) -> Path:
    path = project_split_dir() / file_name
    if path.exists():
        return path
    return _write_split(path, class_groups)
