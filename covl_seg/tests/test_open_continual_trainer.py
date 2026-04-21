import json

import pytest

from covl_seg.engine.open_continual_trainer import OpenContinualTrainer


def _write_mock_config(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
EXPERIMENT:
  DATASET: ade20k_15
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return cfg


def test_trainer_writes_plan_state_and_metrics(tmp_path, capsys):
    cfg = _write_mock_config(tmp_path)
    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run",
        engine="mock",
        seed=0,
        method_name="covl",
        clip_finetune="attention",
        task_spec=None,
        num_tasks=2,
        classes_per_task=2,
        task_seed=0,
        n_pre=1,
        n_main=1,
        eps_f=0.05,
        t_mem="all",
        mix_ratio=[3, 1],
        m_max_total=100,
        m_max_per_class=10,
        ewc_lambda=10.0,
        ewc_topk=4,
        ewc_iters=10,
        enable_ciba=True,
        enable_ctr=True,
        enable_spectral_ogp=True,
        enable_sacr=True,
    )
    trainer.run()
    capsys.readouterr()
    assert (tmp_path / "run" / "task_plan.json").exists()
    assert (tmp_path / "run" / "continual_state.json").exists()
    assert (tmp_path / "run" / "metrics.jsonl").exists()
    state = json.loads((tmp_path / "run" / "continual_state.json").read_text(encoding="utf-8"))
    assert "alpha_star_history" in state
    assert "tau_pred_history" in state
    assert len(state["alpha_star_history"]) >= 1
    metric_lines = (tmp_path / "run" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    first = json.loads(metric_lines[0])
    assert "beta_1_star" in first
    second = json.loads(metric_lines[1])
    assert "ctr_loss" in second
    third = json.loads(metric_lines[2])
    assert "omega_tau_t" in third
    assert "alpha_star" in third


def test_trainer_resume_rejects_method_mismatch(tmp_path):
    cfg = _write_mock_config(tmp_path)
    out = tmp_path / "run"
    out.mkdir(parents=True)
    (out / "continual_state.json").write_text(
        json.dumps({"current_task": 1, "method": "replay"}),
        encoding="utf-8",
    )

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=out,
        engine="mock",
        seed=0,
        method_name="covl",
        clip_finetune="attention",
        task_spec=None,
        num_tasks=1,
        classes_per_task=2,
        task_seed=0,
        n_pre=1,
        n_main=1,
        eps_f=0.05,
        t_mem="all",
        mix_ratio=[3, 1],
        m_max_total=100,
        m_max_per_class=10,
        ewc_lambda=10.0,
        ewc_topk=4,
        ewc_iters=10,
        enable_ciba=True,
        enable_ctr=True,
        enable_spectral_ogp=True,
        enable_sacr=True,
    )
    with pytest.raises(ValueError):
        trainer.run()


def test_trainer_d2_passes_task_and_clip_overrides(tmp_path, monkeypatch, capsys):
    cfg = _write_mock_config(tmp_path)
    calls = []

    def _fake_train(**kwargs):
        calls.append(kwargs)
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback("[d2.utils.events]:  iter: 1  total_loss: 0.1")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**kwargs):
        return {
            "mIoU_all": 10.0,
            "mIoU_old": 9.0,
            "mIoU_new": 11.0,
            "BG-mIoU": 8.0,
            "class_iou_all": {"person": 10.0, "car": 20.0},
            "class_iou_old": {"person": 10.0},
            "class_iou_new": {"car": 20.0},
        }

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2",
        engine="d2",
        seed=0,
        method_name="covl",
        clip_finetune="attention",
        task_spec=None,
        num_tasks=1,
        classes_per_task=2,
        task_seed=0,
        n_pre=1,
        n_main=1,
        eps_f=0.05,
        t_mem="all",
        mix_ratio=[3, 1],
        m_max_total=100,
        m_max_per_class=10,
        ewc_lambda=10.0,
        ewc_topk=4,
        ewc_iters=10,
        enable_ciba=True,
        enable_ctr=True,
        enable_spectral_ogp=True,
        enable_sacr=True,
    )
    trainer.run()
    assert len(calls) == 1
    overrides = calls[0]["extra_overrides"]
    assert "MODEL.SEM_SEG_HEAD.CLIP_FINETUNE" in overrides
    assert "MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES" in overrides
    assert "MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES" in overrides
    assert "SOLVER.MAX_ITER" in overrides
    assert overrides[overrides.index("SOLVER.MAX_ITER") + 1] == "1"
    assert "TEST.EVAL_PERIOD" in overrides
    assert overrides[overrides.index("TEST.EVAL_PERIOD") + 1] == "0"

    metrics_lines = (tmp_path / "run_d2" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    phase1 = json.loads(metrics_lines[0])
    phase2 = json.loads(metrics_lines[1])
    assert phase1["proxy_source"] == "d2_metrics"
    assert phase2["proxy_source"] == "d2_metrics"
    output = capsys.readouterr().out
    assert "continual" in output
    assert "eval start" in output
    assert "eval result" in output
    assert "class IoU detail" in output
    assert "group=old" in output
    assert "group=new" in output


def test_trainer_d2_can_skip_per_task_eval(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    train_calls = []
    eval_calls = []

    def _fake_train(**kwargs):
        train_calls.append(kwargs)
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**kwargs):
        eval_calls.append(kwargs)
        return {"mIoU_all": 10.0, "mIoU_old": 9.0, "mIoU_new": 11.0, "BG-mIoU": 8.0}

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2_skip_eval",
        engine="d2",
        seed=0,
        method_name="covl",
        clip_finetune="attention",
        task_spec=None,
        num_tasks=1,
        classes_per_task=2,
        task_seed=0,
        n_pre=1,
        n_main=1,
        eps_f=0.05,
        t_mem="all",
        mix_ratio=[3, 1],
        m_max_total=100,
        m_max_per_class=10,
        ewc_lambda=10.0,
        ewc_topk=4,
        ewc_iters=10,
        enable_ciba=True,
        enable_ctr=True,
        enable_spectral_ogp=True,
        enable_sacr=True,
        skip_per_task_eval=True,
    )
    trainer.run()
    assert len(train_calls) == 1
    assert len(eval_calls) == 0


def test_d2_task_progress_eval_mode_has_no_train_bar():
    from covl_seg.engine.open_continual_trainer import _D2TaskProgress

    progress = _D2TaskProgress(task_id=1, total_tasks=2, task_total_iters=1, show_train=False)
    try:
        progress.train_callback("iter: 1")
        assert progress._train_pbar is None
    finally:
        progress.close()


def test_trainer_defaults_to_single_task_when_partition_args_absent(tmp_path):
    cfg = _write_mock_config(tmp_path)
    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_defaults",
        engine="mock",
        seed=0,
        method_name="covl",
        clip_finetune="attention",
        task_spec=None,
        num_tasks=None,
        classes_per_task=None,
        task_seed=0,
        n_pre=1,
        n_main=1,
        eps_f=0.05,
        t_mem="all",
        mix_ratio=[3, 1],
        m_max_total=100,
        m_max_per_class=10,
        ewc_lambda=10.0,
        ewc_topk=4,
        ewc_iters=10,
        enable_ciba=True,
        enable_ctr=True,
        enable_spectral_ogp=True,
        enable_sacr=True,
    )

    result = trainer.run(max_tasks=1)
    assert result["tasks_executed"] == 1.0
    task_plan = json.loads((tmp_path / "run_defaults" / "task_plan.json").read_text(encoding="utf-8"))
    assert len(task_plan["tasks"]) == 1
