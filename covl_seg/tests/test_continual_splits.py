import json


def test_split_union_coverage_and_no_duplicates(tmp_path):
    from covl_seg.data.continual_loader import validate_split_mapping

    split = {
        "tasks": [
            {"task_id": 0, "classes": [0, 1, 2]},
            {"task_id": 1, "classes": [3, 4]},
            {"task_id": 2, "classes": [5]},
        ]
    }
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps(split), encoding="utf-8")

    classes = validate_split_mapping(split_file, expected_num_classes=6)
    assert classes == [0, 1, 2, 3, 4, 5]


def test_continual_loader_uses_configured_mix_ratio():
    from covl_seg.data.continual_loader import build_mixed_batch

    task_samples = [f"t{i}" for i in range(12)]
    replay_samples = [f"r{i}" for i in range(6)]

    mixed = build_mixed_batch(
        task_samples=task_samples,
        replay_samples=replay_samples,
        batch_size=8,
        task_to_replay_ratio=(3, 1),
    )
    assert len(mixed["task"]) == 6
    assert len(mixed["replay"]) == 2


def test_builtin_split_files_are_valid():
    from covl_seg.data.ade20k_15 import split_path as ade15_split
    from covl_seg.data.ade20k_100 import split_path as ade100_split
    from covl_seg.data.coco_stuff_164_10 import split_path as coco_split
    from covl_seg.data.continual_loader import validate_split_mapping

    assert len(validate_split_mapping(ade15_split(), expected_num_classes=150)) == 150
    assert len(validate_split_mapping(ade100_split(), expected_num_classes=150)) == 150
    assert len(validate_split_mapping(coco_split(), expected_num_classes=164)) == 164
