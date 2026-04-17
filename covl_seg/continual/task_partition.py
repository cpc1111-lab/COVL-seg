from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class TaskDef:
    task_id: int
    new_classes: List[int]
    seen_classes: List[int]
    background_classes: List[int]


@dataclass(frozen=True)
class TaskPlan:
    tasks: List[TaskDef]

    def to_dict(self) -> dict:
        return {"tasks": [asdict(task) for task in self.tasks]}


def _build_from_new_class_groups(groups: List[List[int]], all_classes: List[int]) -> TaskPlan:
    tasks: List[TaskDef] = []
    seen: List[int] = []
    all_set = set(all_classes)
    for i, new_classes in enumerate(groups, start=1):
        seen = sorted(set(seen).union(new_classes))
        background = sorted(all_set.difference(seen))
        tasks.append(
            TaskDef(
                task_id=i,
                new_classes=sorted(new_classes),
                seen_classes=seen,
                background_classes=background,
            )
        )
    return TaskPlan(tasks=tasks)


def _build_explicit_plan(task_spec: str, all_classes: List[int]) -> TaskPlan:
    payload = json.loads(Path(task_spec).read_text(encoding="utf-8"))
    raw_tasks = payload.get("tasks", [])
    if not raw_tasks:
        raise ValueError("task-spec must contain non-empty 'tasks'")
    groups = []
    for task in raw_tasks:
        new_classes = task.get("new_classes")
        if not isinstance(new_classes, list) or not new_classes:
            raise ValueError("each task must define non-empty 'new_classes' list")
        groups.append([int(x) for x in new_classes])
    return _build_from_new_class_groups(groups=groups, all_classes=all_classes)


def _build_auto_plan(
    num_tasks: int,
    classes_per_task: int,
    all_classes: List[int],
    seed: int,
) -> TaskPlan:
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive")
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be positive")
    classes = list(all_classes)
    rng = random.Random(seed)
    rng.shuffle(classes)
    groups: List[List[int]] = []
    cursor = 0
    for _ in range(num_tasks):
        group = classes[cursor : cursor + classes_per_task]
        if not group:
            break
        groups.append(sorted(group))
        cursor += classes_per_task
    if not groups:
        raise ValueError("auto partition generated no tasks")
    return _build_from_new_class_groups(groups=groups, all_classes=all_classes)


def build_task_plan(
    task_spec: Optional[str],
    num_tasks: Optional[int],
    classes_per_task: Optional[int],
    all_classes: List[int],
    seed: int,
) -> TaskPlan:
    if not all_classes:
        raise ValueError("all_classes must not be empty")
    if task_spec:
        return _build_explicit_plan(task_spec=task_spec, all_classes=all_classes)
    if num_tasks is None or classes_per_task is None:
        raise ValueError("num_tasks and classes_per_task are required when task_spec is absent")
    return _build_auto_plan(
        num_tasks=num_tasks,
        classes_per_task=classes_per_task,
        all_classes=all_classes,
        seed=seed,
    )
