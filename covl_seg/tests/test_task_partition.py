import json

from covl_seg.continual.task_partition import build_task_plan


def test_build_task_plan_from_spec(tmp_path):
    spec = {
        "tasks": [
            {"task_id": 1, "new_classes": [0, 1]},
            {"task_id": 2, "new_classes": [2, 3]},
        ]
    }
    p = tmp_path / "tasks.json"
    p.write_text(json.dumps(spec), encoding="utf-8")

    plan = build_task_plan(
        task_spec=str(p),
        num_tasks=None,
        classes_per_task=None,
        all_classes=[0, 1, 2, 3],
        seed=0,
    )
    assert plan.tasks[0].new_classes == [0, 1]
    assert plan.tasks[1].seen_classes == [0, 1, 2, 3]
    assert plan.tasks[0].background_classes == [2, 3]


def test_build_task_plan_auto_partition_is_deterministic():
    plan_a = build_task_plan(
        task_spec=None,
        num_tasks=2,
        classes_per_task=2,
        all_classes=[0, 1, 2, 3],
        seed=11,
    )
    plan_b = build_task_plan(
        task_spec=None,
        num_tasks=2,
        classes_per_task=2,
        all_classes=[0, 1, 2, 3],
        seed=11,
    )
    assert plan_a.to_dict() == plan_b.to_dict()
