import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def validate_split_mapping(split_file: Path, expected_num_classes: int) -> List[int]:
    payload = json.loads(split_file.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    if not tasks:
        raise ValueError("split file must contain non-empty 'tasks'")

    flat: List[int] = []
    for task in tasks:
        classes = task.get("classes", [])
        if not classes:
            raise ValueError("every task must contain non-empty 'classes'")
        flat.extend(int(x) for x in classes)

    if len(flat) != len(set(flat)):
        raise ValueError("duplicate class id found across tasks")

    sorted_classes = sorted(flat)
    if len(sorted_classes) != expected_num_classes:
        raise ValueError("split class count does not match expected_num_classes")

    if sorted_classes != list(range(expected_num_classes)):
        raise ValueError("split classes must cover contiguous range [0, expected_num_classes)")
    return sorted_classes


def build_mixed_batch(
    task_samples: Sequence[str],
    replay_samples: Sequence[str],
    batch_size: int,
    task_to_replay_ratio: Tuple[int, int] = (3, 1),
) -> Dict[str, List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    task_weight, replay_weight = task_to_replay_ratio
    if task_weight <= 0 or replay_weight <= 0:
        raise ValueError("ratio entries must be positive")

    total_weight = task_weight + replay_weight
    task_count = int(round(batch_size * (task_weight / total_weight)))
    task_count = max(1, min(batch_size - 1, task_count))
    replay_count = batch_size - task_count

    if len(task_samples) < task_count:
        raise ValueError("not enough task samples to construct batch")
    if len(replay_samples) < replay_count:
        raise ValueError("not enough replay samples to construct batch")

    return {
        "task": list(task_samples[:task_count]),
        "replay": list(replay_samples[:replay_count]),
    }
