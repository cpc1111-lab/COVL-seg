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
    )

    payload = run_eval_once(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=out_dir,
        resume_task=1,
        checkpoint=None,
        open_vocab=True,
    )
    assert "mIoU_all" in payload
    assert "ov_mIoU_pc59" in payload
    assert (out_dir / "eval_summary.json").exists()
