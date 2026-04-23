from pathlib import Path


def test_train_once_runs_four_phase_and_writes_metrics(tmp_path: Path):
    from covl_seg.scripts.train_continual import run_train_once

    out_dir = tmp_path / "train"
    result = run_train_once(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        seed=0,
        resume_task=0,
        max_tasks=2,
        engine="mock",
    )

    assert result["num_tasks"] == 2
    assert result["num_phase_records"] == 8
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "checkpoint_task_002.json").exists()


def test_eval_once_reads_train_artifacts_and_writes_summary(tmp_path: Path):
    from covl_seg.scripts.eval_continual import run_eval_once
    from covl_seg.scripts.train_continual import run_train_once

    out_dir = tmp_path / "exp"
    run_train_once(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        seed=0,
        resume_task=0,
        max_tasks=1,
        engine="mock",
    )

    payload = run_eval_once(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        resume_task=1,
        checkpoint=None,
        open_vocab=True,
        engine="mock",
    )
    assert "mIoU_all" in payload
    assert "ov_mIoU_pc59" in payload
    assert (out_dir / "eval_summary.json").exists()


def test_mock_train_resume_task_continues_without_index_error(tmp_path: Path):
    from covl_seg.scripts.train_continual import run_train_once

    out_dir = tmp_path / "resume"
    first = run_train_once(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        seed=0,
        resume_task=0,
        max_tasks=1,
        engine="mock",
    )
    assert first["last_task"] == 1

    resumed = run_train_once(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        seed=0,
        resume_task=1,
        max_tasks=1,
        engine="mock",
    )
    assert resumed["last_task"] == 2
    assert (out_dir / "checkpoint_task_002.pt").exists()


def test_train_once_d2_auto_bootstraps_missing_datasets(monkeypatch, tmp_path: Path):
    from covl_seg.scripts import train_continual as train

    bootstrap_calls = []

    def _fake_bootstrap(*, datasets_root, runtime_root, force_download=False):
        bootstrap_calls.append((datasets_root, runtime_root, force_download))

    def _fake_run_detectron2_train(**_kwargs):
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    monkeypatch.setattr(train, "detectron2_available", lambda: True)
    monkeypatch.setattr(train, "ensure_coco_stuff_ready_for_training", _fake_bootstrap)
    monkeypatch.setattr(train, "run_detectron2_train", _fake_run_detectron2_train)

    out_dir = tmp_path / "d2"
    payload = train.run_train_once(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        seed=0,
        resume_task=0,
        max_tasks=1,
        engine="d2",
        seg_net="vitb",
        datasets_root=str(tmp_path / "datasets"),
    )

    assert payload["last_task"] == 1
    assert len(bootstrap_calls) == 1
    assert str(bootstrap_calls[0][0]).endswith("datasets")
