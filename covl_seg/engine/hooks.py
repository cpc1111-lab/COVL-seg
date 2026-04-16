import json
from pathlib import Path
from typing import Dict


def append_metrics_jsonl(file_path: Path, record: Dict[str, float]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
