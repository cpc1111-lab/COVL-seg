import json
from pathlib import Path

import torch


def test_run_detectron2_train_invokes_catseg_train_net(monkeypatch, tmp_path: Path):
    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        calls.append(
            {
                "cmd": cmd,
                "cwd": cwd,
                "check": check,
                "env": env,
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
    assert any(token.endswith("configs/vitb_384.yaml") for token in cmd)
    assert "SOLVER.AMP.ENABLED" in cmd
    assert cmd[cmd.index("SOLVER.AMP.ENABLED") + 1] == "False"
    assert "SOLVER.BASE_LR" in cmd
    assert cmd[cmd.index("SOLVER.BASE_LR") + 1] == "0.0001"
    assert calls[0]["env"]["DETECTRON2_DATASETS"] == str(d2_runner._workspace_root() / "datasets")
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

    def _fake_run(cmd, cwd, check, env, **_kwargs):
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
        config_path="covl_seg/configs/covl_seg_vitl_ade15.yaml",
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
        assert any(token.endswith("configs/vitl_336.yaml") for token in cmd)
    assert payload["mIoU_all"] == 37.5
    assert payload["mIoU_old"] == 44.0
    assert payload["mIoU_new"] == 21.0
    assert payload["BG-mIoU"] == 12.0
    assert payload["ov_mIoU_pc59"] == 18.0
    assert payload["ov_mIoU_pc459"] == 7.0
    assert payload["ov_mIoU_voc20"] == 52.0
    assert (out_dir / "eval_summary.json").exists()


def test_run_detectron2_eval_uses_coco_dataset_for_coco_config(monkeypatch, tmp_path: Path):
    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        calls.append(cmd)

        output_dir = Path(cmd[cmd.index("OUTPUT_DIR") + 1])
        inference_dir = output_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)

        if '("coco_2017_test_stuff_all_sem_seg",)' in cmd:
            payload = {
                "sem_seg": {
                    "mIoU": 41.5,
                    "seen_IoU": 43.0,
                    "unseen_IoU": 22.0,
                    "IoU-background": 9.0,
                }
            }
        else:
            payload = {"sem_seg": {"mIoU": 0.0}}
        torch.save(payload, inference_dir / "sem_seg_evaluation.pth")

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr(d2_runner, "detectron2_available", lambda: True)
    monkeypatch.setattr(d2_runner.subprocess, "run", _fake_run)

    out_dir = tmp_path / "eval_coco"
    payload = d2_runner.run_detectron2_eval(
        config_path="covl_seg/configs/covl_seg_vitb_coco.yaml",
        output_dir=out_dir,
        resume_task=0,
        checkpoint="/tmp/model_final.pth",
        open_vocab=False,
    )

    assert len(calls) == 1
    cmd = calls[0]
    assert '("coco_2017_test_stuff_all_sem_seg",)' in cmd
    assert "MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON" in cmd
    assert cmd[cmd.index("MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON") + 1] == "datasets/coco.json"
    assert payload["mIoU_all"] == 41.5


def test_run_detectron2_eval_derives_old_new_bg_metrics_from_task_splits(monkeypatch, tmp_path: Path):
    from covl_seg.engine import detectron2_runner as d2_runner

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        output_dir = Path(cmd[cmd.index("OUTPUT_DIR") + 1])
        inference_dir = output_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "sem_seg": {
                "mIoU": 30.0,
                "IoU-person": 50.0,
                "IoU-bicycle": 30.0,
                "IoU-car": 10.0,
            }
        }
        torch.save(payload, inference_dir / "sem_seg_evaluation.pth")

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    splits = tmp_path / "eval_with_splits" / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "seen_indexes.json").write_text("[0, 1]", encoding="utf-8")
    (splits / "new_indexes.json").write_text("[1]", encoding="utf-8")
    (splits / "unseen_indexes.json").write_text("[2]", encoding="utf-8")

    monkeypatch.setattr(d2_runner, "detectron2_available", lambda: True)
    monkeypatch.setattr(d2_runner.subprocess, "run", _fake_run)

    payload = d2_runner.run_detectron2_eval(
        config_path="covl_seg/configs/covl_seg_vitb_coco.yaml",
        output_dir=tmp_path / "eval_with_splits",
        resume_task=1,
        checkpoint="/tmp/model_final.pth",
        open_vocab=False,
    )

    assert payload["mIoU_old"] == 50.0
    assert payload["mIoU_new"] == 30.0
    assert payload["BG-mIoU"] == 10.0
    assert payload["class_iou_old"] == {"person": 50.0}
    assert payload["class_iou_new"] == {"bicycle": 30.0}
    assert payload["class_iou_bg"] == {"car": 10.0}


def test_run_detectron2_eval_can_disable_sliding_window(monkeypatch, tmp_path: Path):
    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        calls.append(cmd)

        output_dir = Path(cmd[cmd.index("OUTPUT_DIR") + 1])
        inference_dir = output_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"sem_seg": {"mIoU": 12.0}}, inference_dir / "sem_seg_evaluation.pth")

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr(d2_runner, "detectron2_available", lambda: True)
    monkeypatch.setattr(d2_runner.subprocess, "run", _fake_run)

    out_dir = tmp_path / "eval_no_slide"
    d2_runner.run_detectron2_eval(
        config_path="covl_seg/configs/covl_seg_vitb_coco.yaml",
        output_dir=out_dir,
        resume_task=0,
        checkpoint="/tmp/model_final.pth",
        open_vocab=False,
        eval_sliding_window=False,
    )

    assert len(calls) == 1
    cmd = calls[0]
    assert "TEST.SLIDING_WINDOW" in cmd
    assert cmd[cmd.index("TEST.SLIDING_WINDOW") + 1] == "False"


def test_run_detectron2_eval_can_limit_eval_samples(monkeypatch, tmp_path: Path):
    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        calls.append({"cmd": cmd, "env": env})

        output_dir = Path(cmd[cmd.index("OUTPUT_DIR") + 1])
        inference_dir = output_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"sem_seg": {"mIoU": 25.0}}, inference_dir / "sem_seg_evaluation.pth")

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    datasets_root = tmp_path / "datasets"
    image_dir = datasets_root / "coco-stuff" / "images" / "val2017"
    ann_dir = datasets_root / "coco-stuff" / "annotations_detectron2" / "val2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(3):
        stem = f"sample_{idx:03d}"
        (image_dir / f"{stem}.jpg").write_text("img", encoding="utf-8")
        (ann_dir / f"{stem}.png").write_text("ann", encoding="utf-8")

    monkeypatch.setenv("DETECTRON2_DATASETS", str(datasets_root))
    monkeypatch.setattr(d2_runner, "detectron2_available", lambda: True)
    monkeypatch.setattr(d2_runner.subprocess, "run", _fake_run)

    out_dir = tmp_path / "eval_subset"
    payload = d2_runner.run_detectron2_eval(
        config_path="covl_seg/configs/covl_seg_vitb_coco.yaml",
        output_dir=out_dir,
        resume_task=0,
        checkpoint="/tmp/model_final.pth",
        open_vocab=False,
        eval_max_samples=2,
    )

    assert payload["mIoU_all"] == 25.0
    assert len(calls) == 1
    subset_root = Path(calls[0]["env"]["DETECTRON2_DATASETS"])
    subset_images = list((subset_root / "coco-stuff" / "images" / "val2017").glob("*.jpg"))
    subset_anns = list((subset_root / "coco-stuff" / "annotations_detectron2" / "val2017").glob("*.png"))
    assert len(subset_images) == 2
    assert len(subset_anns) == 2


def test_run_detectron2_eval_subset_env_root_is_absolute_for_relative_output_dir(
    monkeypatch,
    tmp_path: Path,
):
    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        calls.append({"cmd": cmd, "env": env})

        output_dir = Path(cmd[cmd.index("OUTPUT_DIR") + 1])
        inference_dir = output_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"sem_seg": {"mIoU": 30.0}}, inference_dir / "sem_seg_evaluation.pth")

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    datasets_root = tmp_path / "datasets"
    image_dir = datasets_root / "coco-stuff" / "images" / "val2017"
    ann_dir = datasets_root / "coco-stuff" / "annotations_detectron2" / "val2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "sample_000.jpg").write_text("img", encoding="utf-8")
    (ann_dir / "sample_000.png").write_text("ann", encoding="utf-8")

    monkeypatch.setenv("DETECTRON2_DATASETS", str(datasets_root))
    monkeypatch.setattr(d2_runner, "detectron2_available", lambda: True)
    monkeypatch.setattr(d2_runner.subprocess, "run", _fake_run)

    relative_output = Path("work_dirs") / "relative_eval"
    d2_runner.run_detectron2_eval(
        config_path="covl_seg/configs/covl_seg_vitb_coco.yaml",
        output_dir=relative_output,
        resume_task=0,
        checkpoint="/tmp/model_final.pth",
        open_vocab=False,
        eval_max_samples=1,
    )

    assert len(calls) == 1
    subset_root = Path(calls[0]["env"]["DETECTRON2_DATASETS"])
    assert subset_root.is_absolute()


def test_run_detectron2_eval_uses_absolute_output_dir_for_relative_input(monkeypatch, tmp_path: Path):
    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        calls.append(cmd)

        output_dir = Path(cmd[cmd.index("OUTPUT_DIR") + 1])
        inference_dir = output_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"sem_seg": {"mIoU": 19.0}}, inference_dir / "sem_seg_evaluation.pth")

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    datasets_root = tmp_path / "datasets"
    image_dir = datasets_root / "coco-stuff" / "images" / "val2017"
    ann_dir = datasets_root / "coco-stuff" / "annotations_detectron2" / "val2017"
    image_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "sample_000.jpg").write_text("img", encoding="utf-8")
    (ann_dir / "sample_000.png").write_text("ann", encoding="utf-8")

    monkeypatch.setenv("DETECTRON2_DATASETS", str(datasets_root))
    monkeypatch.setattr(d2_runner, "detectron2_available", lambda: True)
    monkeypatch.setattr(d2_runner.subprocess, "run", _fake_run)

    payload = d2_runner.run_detectron2_eval(
        config_path="covl_seg/configs/covl_seg_vitb_coco.yaml",
        output_dir=Path("work_dirs") / "relative_eval_abs",
        resume_task=0,
        checkpoint="/tmp/model_final.pth",
        open_vocab=False,
    )

    assert payload["mIoU_all"] == 19.0
    assert len(calls) == 1
    output_dir_arg = Path(calls[0][calls[0].index("OUTPUT_DIR") + 1])
    assert output_dir_arg.is_absolute()


def test_run_detectron2_train_retries_with_low_mem_overrides_on_oom(monkeypatch, tmp_path: Path):
    import subprocess

    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        calls.append({"cmd": cmd, "env": env})
        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                output="",
                stderr="torch.cuda.OutOfMemoryError: CUDA out of memory",
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
        json.dumps({"iteration": 0, "total_loss": 1.0}) + "\n",
        encoding="utf-8",
    )

    result = d2_runner.run_detectron2_train(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        seed=0,
        resume_task=0,
        max_tasks=1,
    )

    assert result["num_tasks"] == 1
    assert len(calls) == 2
    retry_cmd = calls[1]["cmd"]
    assert "SOLVER.IMS_PER_BATCH" in retry_cmd
    assert retry_cmd[retry_cmd.index("SOLVER.IMS_PER_BATCH") + 1] == "1"
    assert "SOLVER.AMP.ENABLED" in retry_cmd
    assert retry_cmd[retry_cmd.index("SOLVER.AMP.ENABLED") + 1] == "False"
    assert "SOLVER.BASE_LR" in retry_cmd
    assert retry_cmd[retry_cmd.index("SOLVER.BASE_LR") + 1] == "0.0001"
    assert "INPUT.CROP.SIZE" in retry_cmd
    crop_indices = [i for i, token in enumerate(retry_cmd) if token == "INPUT.CROP.SIZE"]
    assert retry_cmd[crop_indices[-1] + 1] == "(320,320)"


def test_run_detectron2_train_accepts_extra_overrides(monkeypatch, tmp_path: Path):
    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        calls.append(cmd)

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr(d2_runner, "detectron2_available", lambda: True)
    monkeypatch.setattr(d2_runner.subprocess, "run", _fake_run)

    out_dir = tmp_path / "exp"
    (out_dir / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps({"iteration": 0, "total_loss": 1.0}) + "\n",
        encoding="utf-8",
    )

    d2_runner.run_detectron2_train(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        seed=0,
        resume_task=0,
        max_tasks=1,
        extra_overrides=["MODEL.SEM_SEG_HEAD.CLIP_FINETUNE", "attention"],
    )

    assert len(calls) == 1
    cmd = calls[0]
    assert "MODEL.SEM_SEG_HEAD.CLIP_FINETUNE" in cmd
    assert cmd[cmd.index("MODEL.SEM_SEG_HEAD.CLIP_FINETUNE") + 1] == "attention"


def test_run_detectron2_train_retries_with_stability_overrides_on_nan(monkeypatch, tmp_path: Path):
    import subprocess

    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        calls.append({"cmd": cmd, "env": env})
        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                output="",
                stderr="FloatingPointError: Loss became infinite or NaN at iteration=7959!\nloss_dict = {'loss_sem_seg': nan}",
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
        json.dumps({"iteration": 0, "total_loss": 1.0}) + "\n",
        encoding="utf-8",
    )

    result = d2_runner.run_detectron2_train(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        seed=0,
        resume_task=0,
        max_tasks=1,
    )

    assert result["num_tasks"] == 1
    assert len(calls) == 2
    retry_cmd = calls[1]["cmd"]
    assert "SOLVER.AMP.ENABLED" in retry_cmd
    assert retry_cmd[retry_cmd.index("SOLVER.AMP.ENABLED") + 1] == "False"
    assert "SOLVER.BASE_LR" in retry_cmd
    assert retry_cmd[retry_cmd.index("SOLVER.BASE_LR") + 1] == "0.0001"


def test_run_detectron2_train_uses_custom_project_root(monkeypatch, tmp_path: Path):
    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        calls.append({"cmd": cmd, "cwd": cwd})

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    project_root = tmp_path / "seg_project"
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "configs" / "vitb_384.yaml").write_text("MODEL: {}\n", encoding="utf-8")

    monkeypatch.setenv("COVL_SEG_D2_PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(d2_runner, "detectron2_available", lambda: True)
    monkeypatch.setattr(d2_runner.subprocess, "run", _fake_run)

    out_dir = tmp_path / "exp"
    (out_dir / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"iteration": 0, "total_loss": 1.0}) + "\n", encoding="utf-8")

    d2_runner.run_detectron2_train(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        seed=0,
        resume_task=0,
        max_tasks=1,
    )

    assert len(calls) == 1
    assert calls[0]["cwd"] == str(project_root)
    assert any(token == str(project_root / "configs" / "vitb_384.yaml") for token in calls[0]["cmd"])


def test_run_detectron2_train_supports_segmentation_network_preset(monkeypatch, tmp_path: Path):
    from covl_seg.engine import detectron2_runner as d2_runner

    calls = []

    def _fake_run(cmd, cwd, check, env, **_kwargs):
        calls.append(cmd)

        class _Completed:
            stdout = ""
            stderr = ""

        return _Completed()

    project_root = tmp_path / "seg_project"
    (project_root / "configs").mkdir(parents=True, exist_ok=True)
    (project_root / "configs" / "swin_b_384.yaml").write_text("MODEL: {}\n", encoding="utf-8")

    monkeypatch.setenv("COVL_SEG_D2_PROJECT_ROOT", str(project_root))
    monkeypatch.setattr(d2_runner, "detectron2_available", lambda: True)
    monkeypatch.setattr(d2_runner.subprocess, "run", _fake_run)

    out_dir = tmp_path / "exp"
    (out_dir / "metrics.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({"iteration": 0, "total_loss": 1.0}) + "\n", encoding="utf-8")

    d2_runner.run_detectron2_train(
        config_path="swin_b",
        output_dir=out_dir,
        seed=1,
        resume_task=0,
        max_tasks=1,
    )

    assert len(calls) == 1
    assert any(token == str(project_root / "configs" / "swin_b_384.yaml") for token in calls[0])


def test_default_project_root_is_internal_vendor(monkeypatch):
    from covl_seg.engine import detectron2_runner as d2_runner

    monkeypatch.delenv("COVL_SEG_D2_PROJECT_ROOT", raising=False)
    resolved = d2_runner._d2_project_root()

    expected = d2_runner._workspace_root() / "covl_seg" / "vendor" / "covl_seg_d2_runtime"
    assert resolved == expected
