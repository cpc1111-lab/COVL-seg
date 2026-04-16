import json


def test_metrics_jsonl_append(tmp_path):
    from covl_seg.engine.hooks import append_metrics_jsonl

    out_file = tmp_path / "metrics.jsonl"
    append_metrics_jsonl(out_file, {"task": 1, "mIoU_all": 21.5})
    append_metrics_jsonl(out_file, {"task": 2, "mIoU_all": 22.7})

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["task"] == 1
    assert second["mIoU_all"] == 22.7
