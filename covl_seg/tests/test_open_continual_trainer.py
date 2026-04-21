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
    output = capsys.readouterr().out
    assert "task 1/" in output
    assert "remaining_task_iters" in output
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
        return {"mIoU_all": 10.0, "mIoU_old": float("nan")}

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

    metrics_lines = (tmp_path / "run_d2" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    phase1 = json.loads(metrics_lines[0])
    phase2 = json.loads(metrics_lines[1])
    eval_record = next(m for m in (json.loads(line) for line in metrics_lines) if m.get("phase") == "eval")
    assert phase1["proxy_source"] == "d2_metrics"
    assert phase2["proxy_source"] == "d2_metrics"
    assert eval_record["mIoU_all"] == 10.0
    assert eval_record["mIoU_old"] is None
    assert eval_record["mIoU_new"] is None
    assert eval_record["BG-mIoU"] is None
    assert "NaN" not in "\n".join(metrics_lines)
    output = capsys.readouterr().out
    assert "continual" in output
    assert "eval start" in output
    assert "eval result" in output


def test_trainer_balanced_controller_logs_metrics_and_persists_state(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)

    def _fake_train(**kwargs):
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**kwargs):
        task_id = int(kwargs["resume_task"])
        by_task = {
            1: {"mIoU_all": 10.0, "mIoU_old": 9.0, "mIoU_new": 11.0, "BG-mIoU": 8.0},
            2: {"mIoU_all": 8.0, "mIoU_old": 7.0, "mIoU_new": 9.0, "BG-mIoU": 6.0},
        }
        return by_task[task_id]

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_balanced",
        engine="d2",
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
        balanced_profile="balanced",
    )
    trainer.run()

    metrics_lines = (tmp_path / "run_balanced" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    metrics = [json.loads(line) for line in metrics_lines]
    ctrl_records = [m for m in metrics if m.get("phase") == "balanced_ctrl"]
    phase4_records = [m for m in metrics if m.get("phase") == "phase4"]
    assert len(ctrl_records) == 2
    assert len(phase4_records) == 2
    assert "delta_new" in ctrl_records[0]
    assert "delta_old" in ctrl_records[0]
    assert "delta_all" in ctrl_records[0]
    assert "ov_min_delta" in ctrl_records[0]
    assert "alpha_floor" in ctrl_records[0]
    assert "ov_guard_triggered" in ctrl_records[0]
    assert "ov_guard_state" in ctrl_records[0]
    assert ctrl_records[1]["ov_guard_triggered"] is True

    assert "replay_rho_new" in phase4_records[1]
    assert "replay_rho_old" in phase4_records[1]
    assert "replay_priority_new_term" in phase4_records[1]
    assert "replay_priority_old_term" in phase4_records[1]
    assert phase4_records[1]["replay_rho_new"] == 0.0
    assert phase4_records[1]["replay_priority_new_term"] == 0.0

    state = json.loads((tmp_path / "run_balanced" / "continual_state.json").read_text(encoding="utf-8"))
    assert "balanced_controller" in state
    assert "balanced_prev_eval" in state
    assert "alpha_floor" in state["balanced_controller"]
    assert "g_stab" in state["balanced_controller"]
    assert "rho_old" in state["balanced_controller"]
    assert "rho_new" in state["balanced_controller"]
    assert "w_ctr" in state["balanced_controller"]
    assert "ov_guard_triggered" in state["balanced_controller"]
    assert "ov_guard_state" in state["balanced_controller"]


def test_trainer_balanced_controller_does_not_drift_without_eval_signals(tmp_path):
    cfg = _write_mock_config(tmp_path)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_balanced_mock",
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
        balanced_profile="balanced",
    )
    trainer.run()

    metrics_text = (tmp_path / "run_balanced_mock" / "metrics.jsonl").read_text(encoding="utf-8")
    assert "NaN" not in metrics_text
    metrics = [json.loads(line) for line in metrics_text.strip().splitlines()]
    ctrl_records = [m for m in metrics if m.get("phase") == "balanced_ctrl"]
    assert len(ctrl_records) == 2
    assert ctrl_records[0]["delta_new"] is None
    assert ctrl_records[0]["delta_old"] is None
    assert ctrl_records[0]["delta_all"] is None
    assert ctrl_records[0]["ov_min_delta"] is None

    state_text = (tmp_path / "run_balanced_mock" / "continual_state.json").read_text(encoding="utf-8")
    assert "NaN" not in state_text
    state = json.loads(state_text)
    controller = state["balanced_controller"]
    assert controller["alpha_floor"] == 0.0
    assert controller["g_stab"] == 0.0
    assert controller["rho_old"] == 0.0
    assert controller["rho_new"] == 0.0
    assert controller["w_ctr"] == 0.0
    assert controller["ov_guard_triggered"] is False
    assert controller["ov_guard_state"] is False
