import json
from pathlib import Path

import torch


def test_run_detectron2_train_invokes_catseg_train_net(monkeypatch, tmp_path: Path):
    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, capture_output, text):
        calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "check": check,
                "capture_output": capture_output,
                "text": text,
            }
        )

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr(d2_runner, "detectron2_available", lambda: True)
    monkeypatch.setattr(d2_runner.subprocess, "run", _fake_run)

    out_dir = tmp_path / "exp"
    (out_dir / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        "\n".join(
            [
                json.dumps({"iteration": 0, "total_loss": 1.2}),
                json.dumps({"iteration": 1, "total_loss": 0.8}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = d2_runner.run_detectron2_train(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        seed=3,
        resume_task=1,
        max_tasks=2,
    )

    assert result["num_tasks"] == 2
    assert result["last_task"] == 3
    assert result["num_phase_records"] == 2
    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    assert cmd[0] == "python"
    assert cmd[1] == "train_net.py"
    assert "--config-file" in cmd
    assert "--num-gpus" in cmd
    assert "OUTPUT_DIR" in cmd
    assert str(out_dir) in cmd
    assert any(token.endswith("CAT-Seg/configs/vitb_384.yaml") for token in cmd)
    metrics_lines = (out_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 2
    first = json.loads(metrics_lines[0])
    second = json.loads(metrics_lines[1])
    assert first["phase"] == "phase1"
    assert second["phase"] == "phase2"
    assert first["task"] == 3.0
    assert second["task"] == 3.0


def test_run_detectron2_eval_invokes_eval_only(monkeypatch, tmp_path: Path):
    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, capture_output, text):
        calls.append(cmd)

        output_dir = Path(cmd[cmd.index("OUTPUT_DIR") + 1])
        inference_dir = output_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)

        if '("ade20k_150_test_sem_seg",)' in cmd:
            payload = {
                "sem_seg": {
                    "mIoU": 37.5,
                    "seen_IoU": 44.0,
                    "unseen_IoU": 21.0,
                    "IoU-background": 12.0,
                }
            }
        elif '("context_59_test_sem_seg",)' in cmd:
            payload = {"sem_seg": {"mIoU": 18.0}}
        elif '("context_459_test_sem_seg",)' in cmd:
            payload = {"sem_seg": {"mIoU": 7.0}}
        elif '("voc_2012_test_sem_seg",)' in cmd:
            payload = {"sem_seg": {"mIoU": 52.0}}
        else:
            payload = {"sem_seg": {"mIoU": 0.0}}
        torch.save(payload, inference_dir / "sem_seg_evaluation.pth")

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr(d2_runner, "detectron2_available", lambda: True)
    monkeypatch.setattr(d2_runner.subprocess, "run", _fake_run)

    out_dir = tmp_path / "eval"

    payload = d2_runner.run_detectron2_eval(
        config_path="CAT-Seg/configs/vitl_336.yaml",
        output_dir=out_dir,
        resume_task=4,
        checkpoint="/tmp/model_final.pth",
        open_vocab=True,
    )

    assert len(calls) == 4
    for cmd in calls:
        assert "--eval-only" in cmd
        assert "MODEL.WEIGHTS" in cmd
        assert "/tmp/model_final.pth" in cmd
        assert any(token.endswith("CAT-Seg/configs/vitl_336.yaml") for token in cmd)
    assert payload["mIoU_all"] == 37.5
    assert payload["mIoU_old"] == 44.0
    assert payload["mIoU_new"] == 21.0
    assert payload["BG-mIoU"] == 12.0
    assert payload["ov_mIoU_pc59"] == 18.0
    assert payload["ov_mIoU_pc459"] == 7.0
    assert payload["ov_mIoU_voc20"] == 52.0
    assert (out_dir / "eval_summary.json").exists()
