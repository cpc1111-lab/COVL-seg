import json
from pathlib import Path

import pytest
import torch

from covl_seg.engine.detectron2_runner import _resolve_experiment_spec
from covl_seg.engine.open_continual_trainer import OpenContinualTrainer, _resolve_task_class_names


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
    assert first["engine"] == "mock"
    checkpoint = json.loads((tmp_path / "run" / "checkpoint_task_002.json").read_text(encoding="utf-8"))
    mock_model_path = checkpoint["mock_model_path"]
    assert Path(mock_model_path).exists()


def test_trainer_mock_resume_restores_saved_model_state(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    out_dir = tmp_path / "run_resume_mock_model"

    first = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=out_dir,
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
    first.run(max_tasks=1)

    checkpoint = json.loads((out_dir / "checkpoint_task_001.json").read_text(encoding="utf-8"))
    mock_model_path = Path(checkpoint["mock_model_path"])
    payload = torch.load(mock_model_path, map_location="cpu")
    state_dict = payload["model_state_dict"]
    param_key = next((k for k in state_dict.keys() if k.startswith("core_model.")), next(iter(state_dict.keys())))
    state_dict[param_key] = torch.full_like(state_dict[param_key], 0.123)
    payload["model_state_dict"] = state_dict
    torch.save(payload, mock_model_path)

    captured = {}

    def _fake_run_mock_task_training(*, model, task, cfg, basis_history):
        captured["task_id"] = task.task_id
        captured["param"] = float(next(model.parameters()).detach().reshape(-1)[0].item())
        next_basis = basis_history[0].detach().cpu().tolist() if basis_history else [0.1]
        return model, {
            "phase1": {"phase": "phase1", "task": float(task.task_id), "loss": 1.0, "beta_1_star": 0.1},
            "phase2": {"phase": "phase2", "task": float(task.task_id), "loss": 2.0, "ctr_loss": 0.2},
            "phase3": {
                "phase": "phase3",
                "task": float(task.task_id),
                "loss": 3.0,
                "alpha_star": 0.3,
                "tau_pred": 0.4,
                "subspace_basis": next_basis,
            },
            "phase4": {"phase": "phase4", "task": float(task.task_id), "loss": 4.0},
        }

    monkeypatch.setattr(
        "covl_seg.engine.open_continual_trainer.run_mock_task_training",
        _fake_run_mock_task_training,
    )

    resumed = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=out_dir,
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
        resume_task=1,
    )
    resumed.run(max_tasks=1)

    assert captured["task_id"] == 2
    assert captured["param"] == pytest.approx(0.123)
    resumed_state = json.loads((out_dir / "continual_state.json").read_text(encoding="utf-8"))
    assert "latest_mock_model_path" in resumed_state


def test_trainer_mock_run_does_not_require_taxonomy_resolution(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)

    def _fail_if_called(_cfg):
        raise AssertionError("mock engine should not resolve class taxonomy")

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", _fail_if_called)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_no_taxonomy",
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

    trainer.run()


def test_trainer_mock_run_uses_non_strict_class_count_inference_for_unknown_dataset(tmp_path):
    cfg = tmp_path / "unknown_dataset.yaml"
    cfg.write_text(
        """
EXPERIMENT:
  DATASET: cityscapes
""".strip()
        + "\n",
        encoding="utf-8",
    )

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_unknown_dataset",
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

    trainer.run()


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


def test_trainer_mock_resume_rejects_missing_state(tmp_path):
    cfg = _write_mock_config(tmp_path)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_missing_state",
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
        resume_task=1,
    )

    with pytest.raises(ValueError, match="Mock resume requires prior continual state"):
        trainer.run()


def test_trainer_mock_resume_rejects_invalid_artifact_path(tmp_path):
    cfg = _write_mock_config(tmp_path)
    out = tmp_path / "run_invalid_mock_resume"
    out.mkdir(parents=True, exist_ok=True)
    (out / "continual_state.json").write_text(
        json.dumps(
            {
                "current_task": 1,
                "method": "covl",
                "latest_mock_model_path": "missing_mock_model.pth",
            }
        ),
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
        resume_task=1,
    )

    with pytest.raises(ValueError, match="valid mock model artifact"):
        trainer.run()


def test_trainer_d2_resume_rejects_missing_prior_model_artifact(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    out = tmp_path / "run_d2_resume"
    out.mkdir(parents=True)
    (out / "continual_state.json").write_text(
        json.dumps(
            {
                "current_task": 1,
                "method": "covl",
                "latest_model_path": str(out / "task_001" / "model_final.pth"),
            }
        ),
        encoding="utf-8",
    )

    def _fake_train(**kwargs):
        task_out = kwargs["output_dir"]
        task_out.mkdir(parents=True, exist_ok=True)
        (task_out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr(
        "covl_seg.engine.open_continual_trainer.run_detectron2_eval",
        lambda **_kwargs: {"mIoU_all": 10.0, "mIoU_old": 9.0, "mIoU_new": 11.0, "BG-mIoU": 8.0},
    )
    monkeypatch.setattr(
        "covl_seg.engine.open_continual_trainer._resolve_task_class_names",
        lambda _cfg: [f"class_{idx}" for idx in range(150)],
    )

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=out,
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
        resume_task=1,
    )

    with pytest.raises(ValueError, match="prior model artifact"):
        trainer.run()


def test_trainer_d2_resume_accepts_expected_task_artifact_when_latest_model_path_is_stale(
    tmp_path,
    monkeypatch,
):
    cfg = _write_mock_config(tmp_path)
    out = tmp_path / "run_d2_resume_fallback"
    expected_weights = out / "task_001" / "model_final.pth"
    expected_weights.parent.mkdir(parents=True, exist_ok=True)
    expected_weights.write_text("weights", encoding="utf-8")
    stale_weights = out / "stale" / "model_final.pth"
    stale_weights.parent.mkdir(parents=True, exist_ok=True)
    stale_weights.write_text("stale-weights", encoding="utf-8")
    (out / "continual_state.json").write_text(
        json.dumps(
            {
                "current_task": 1,
                "method": "covl",
                "latest_model_path": str(stale_weights),
            }
        ),
        encoding="utf-8",
    )

    calls = []

    def _fake_train(**kwargs):
        calls.append(kwargs)
        task_out = kwargs["output_dir"]
        task_out.mkdir(parents=True, exist_ok=True)
        (task_out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr(
        "covl_seg.engine.open_continual_trainer.run_detectron2_eval",
        lambda **_kwargs: {"mIoU_all": 10.0, "mIoU_old": 9.0, "mIoU_new": 11.0, "BG-mIoU": 8.0},
    )
    monkeypatch.setattr(
        "covl_seg.engine.open_continual_trainer._resolve_task_class_names",
        lambda _cfg: [f"class_{idx}" for idx in range(150)],
    )

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=out,
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
        resume_task=1,
    )

    trainer.run()

    assert len(calls) == 1
    overrides = calls[0]["extra_overrides"]
    assert overrides[overrides.index("MODEL.WEIGHTS") + 1] == str(expected_weights)


def test_trainer_d2_passes_task_and_clip_overrides(tmp_path, monkeypatch, capsys):
    cfg = _write_mock_config(tmp_path)
    calls = []
    class_names = [f"class_{idx}" for idx in range(150)]

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
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", lambda _cfg: class_names)

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
    assert "MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON" in overrides
    assert "MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON" in overrides
    assert "MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES" in overrides
    assert "MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES" in overrides
    assert "MODEL.SEM_SEG_HEAD.TRAIN_OLD_CLASS_INDEXES" in overrides
    assert "MODEL.SEM_SEG_HEAD.TRAIN_UNSEEN_CLASS_INDEXES" in overrides
    assert "MODEL.SEM_SEG_HEAD.OLD_TEACHER_WEIGHTS" in overrides
    assert "MODEL.SEM_SEG_HEAD.LAMBDA_OLD_KD" in overrides
    assert "MODEL.SEM_SEG_HEAD.LAMBDA_OLD_CLIP" in overrides
    assert "MODEL.SEM_SEG_HEAD.LAMBDA_UNSEEN_CLIP" in overrides
    assert "SOLVER.MAX_ITER" in overrides
    assert overrides[overrides.index("SOLVER.MAX_ITER") + 1] == "1"
    train_json = tmp_path / "run_d2" / "task_001" / "splits" / "train_class_names.json"
    test_json = tmp_path / "run_d2" / "task_001" / "splits" / "test_class_names.json"
    train_indexes = tmp_path / "run_d2" / "task_001" / "splits" / "seen_indexes.json"
    old_indexes = tmp_path / "run_d2" / "task_001" / "splits" / "old_indexes.json"
    new_indexes = tmp_path / "run_d2" / "task_001" / "splits" / "new_indexes.json"
    test_indexes = tmp_path / "run_d2" / "task_001" / "splits" / "unseen_indexes.json"
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON") + 1] == str(train_json)
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON") + 1] == str(test_json)
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES") + 1] == str(train_indexes)
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES") + 1] == str(test_indexes)
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.TRAIN_OLD_CLASS_INDEXES") + 1] == str(old_indexes)
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.TRAIN_UNSEEN_CLASS_INDEXES") + 1] == str(test_indexes)
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.OLD_TEACHER_WEIGHTS") + 1] == ""
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.LAMBDA_OLD_KD") + 1] == "1.0"
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.LAMBDA_OLD_CLIP") + 1] == "0.1"
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.LAMBDA_UNSEEN_CLIP") + 1] == "0.2"
    assert json.loads(old_indexes.read_text(encoding="utf-8")) == []
    assert len(json.loads(new_indexes.read_text(encoding="utf-8"))) == 2
    assert json.loads(train_json.read_text(encoding="utf-8")) == class_names
    assert json.loads(test_json.read_text(encoding="utf-8")) == class_names

    metrics_lines = (tmp_path / "run_d2" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    metrics = [json.loads(line) for line in metrics_lines]
    eval_record = next(m for m in metrics if m.get("phase") == "eval")
    assert any(m.get("phase") == "phase1" for m in metrics)
    assert any(m.get("phase") == "phase2" for m in metrics)
    assert any(m.get("phase") == "phase3" for m in metrics)
    assert any(m.get("phase") == "phase4" for m in metrics)
    phase_rows = [m for m in metrics if m.get("phase") in {"phase1", "phase2", "phase3", "phase4"}]
    assert phase_rows
    assert all(m.get("proxy_source") == "derived_from_d2_train_metrics" for m in phase_rows)
    assert eval_record["mIoU_all"] == 10.0
    assert eval_record["mIoU_old"] is None
    assert eval_record["mIoU_new"] is None
    assert eval_record["BG-mIoU"] is None
    assert "NaN" not in "\n".join(metrics_lines)
    output = capsys.readouterr().out
    assert "continual" in output
    assert "eval start" in output
    assert "eval result" in output


def test_trainer_d2_passes_continual_runtime_flags(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    calls = []

    def _fake_train(**kwargs):
        calls.append(kwargs)
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", lambda **kwargs: {"mIoU_all": 1.0})
    monkeypatch.setattr(
        "covl_seg.engine.open_continual_trainer._resolve_task_class_names",
        lambda _cfg: [f"class_{idx}" for idx in range(150)],
    )

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2_flags",
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
        ewc_topk=8,
        ewc_iters=10,
        enable_ciba=False,
        enable_ctr=True,
        enable_spectral_ogp=True,
        enable_sacr=True,
        lambda_old_kd=1.5,
        lambda_old_clip=0.3,
        lambda_unseen_clip=0.25,
    )

    trainer.run()

    overrides = calls[0]["extra_overrides"]
    assert overrides[overrides.index("MODEL.COVL.ENABLE_CIBA") + 1] == "False"
    assert overrides[overrides.index("MODEL.COVL.ENABLE_CTR") + 1] == "True"
    assert overrides[overrides.index("MODEL.COVL.ENABLE_OGP") + 1] == "True"
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.LAMBDA_OLD_KD") + 1] == "1.5"
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.LAMBDA_OLD_CLIP") + 1] == "0.3"
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.LAMBDA_UNSEEN_CLIP") + 1] == "0.25"


def test_trainer_d2_logs_real_source_derived_continual_record(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    class_names = [f"class_{idx}" for idx in range(150)]

    def _fake_train(**kwargs):
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(
            '{"iteration":0,"total_loss":1.0,"loss_ctr":0.2}\n',
            encoding="utf-8",
        )
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**_kwargs):
        return {
            "mIoU_all": 10.0,
            "mIoU_old": 9.0,
            "mIoU_new": 11.0,
            "BG-mIoU": 8.0,
            "class_iou_all": {"wall": 0.4, "floor": 0.5},
            "class_iou_old": {"wall": 0.4},
            "class_iou_new": {"floor": 0.5},
            "class_iou_bg": {},
        }

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", lambda _cfg: class_names)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2_real_source",
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

    metrics_lines = (tmp_path / "run_d2_real_source" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    metrics = [json.loads(line) for line in metrics_lines]
    derived_record = next(m for m in metrics if m.get("phase") == "continual_real")
    eval_record = next(m for m in metrics if m.get("phase") == "eval")

    assert derived_record["metric_source"] == "derived_from_real_artifacts"
    assert "beta_1_star" in derived_record
    assert "ctr_loss" in derived_record
    assert "alpha_star" in derived_record
    assert "tau_pred" in derived_record
    assert "omega_tau_t" in derived_record

    assert eval_record["mIoU_all"] == 10.0
    assert eval_record["mIoU_old"] == 9.0
    assert eval_record["mIoU_new"] == 11.0
    assert eval_record["BG-mIoU"] == 8.0


def test_trainer_d2_derived_real_record_keeps_ctr_loss_null_without_source_metric(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    class_names = [f"class_{idx}" for idx in range(150)]

    def _fake_train(**kwargs):
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(
            '{"iteration":0,"total_loss":1.0}\n',
            encoding="utf-8",
        )
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**_kwargs):
        return {
            "mIoU_all": 10.0,
            "mIoU_old": 9.0,
            "mIoU_new": 11.0,
            "BG-mIoU": 8.0,
            "class_iou_all": {"wall": 0.4, "floor": 0.5},
            "class_iou_old": {"wall": 0.4},
            "class_iou_new": {"floor": 0.5},
            "class_iou_bg": {},
        }

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", lambda _cfg: class_names)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2_real_ctr_null",
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

    metrics_lines = (tmp_path / "run_d2_real_ctr_null" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    metrics = [json.loads(line) for line in metrics_lines]
    derived_record = next(m for m in metrics if m.get("phase") == "continual_real")
    assert derived_record["metric_source"] == "derived_from_real_artifacts"
    assert derived_record["ctr_loss"] is None


def test_trainer_d2_summarizes_real_continual_loss_fields(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    class_names = [f"class_{idx}" for idx in range(150)]

    def _fake_train(**kwargs):
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(
            "\n".join(
                [
                    '{"iteration":0,"total_loss":1.0,"loss_sem_seg":1.0,"loss_old_kd":0.2,"loss_old_clip":0.1,"loss_unseen_clip":0.05,"loss_ciba":0.15,"loss_ctr":0.1}',
                    '{"iteration":1,"total_loss":0.9,"loss_sem_seg":0.9,"loss_old_kd":0.1,"loss_old_clip":0.08,"loss_unseen_clip":0.04,"loss_ciba":0.11,"loss_ctr":0.07}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {"num_tasks": 1, "num_phase_records": 2, "last_task": 1}

    def _fake_eval(**_kwargs):
        return {
            "mIoU_all": 10.0,
            "mIoU_old": 9.0,
            "mIoU_new": 11.0,
            "BG-mIoU": 8.0,
            "class_iou_all": {"wall": 0.4, "floor": 0.5},
            "class_iou_old": {"wall": 0.4},
            "class_iou_new": {"floor": 0.5},
            "class_iou_bg": {},
        }

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", lambda _cfg: class_names)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2_task_summary",
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

    metrics_lines = (tmp_path / "run_d2_task_summary" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    metrics = [json.loads(line) for line in metrics_lines]
    task_summary = next(m for m in metrics if m.get("phase") == "task_summary")

    assert task_summary["loss_sem_seg"] == pytest.approx(0.95)
    assert task_summary["loss_old_kd"] == pytest.approx(0.15)
    assert task_summary["loss_old_clip"] == pytest.approx(0.09)
    assert task_summary["loss_unseen_clip"] == pytest.approx(0.045)
    assert task_summary["loss_ciba"] == pytest.approx(0.13)
    assert task_summary["loss_ctr"] == pytest.approx(0.085)


def test_trainer_d2_writes_task_conditioned_class_json_artifacts(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    class_names = [f"class_{idx}" for idx in range(150)]
    calls = []

    def _fake_train(**kwargs):
        calls.append(kwargs)
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**_kwargs):
        return {"mIoU_all": 10.0, "mIoU_old": 9.0, "mIoU_new": 11.0, "BG-mIoU": 8.0}

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", lambda _cfg: class_names)

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

    split_dir = tmp_path / "run_d2" / "task_001" / "splits"
    train_json = split_dir / "train_class_names.json"
    test_json = split_dir / "test_class_names.json"
    assert train_json.exists()
    assert test_json.exists()
    assert json.loads(train_json.read_text(encoding="utf-8")) == class_names
    assert json.loads(test_json.read_text(encoding="utf-8")) == class_names
    overrides = calls[0]["extra_overrides"]
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON") + 1] == str(train_json)
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON") + 1] == str(test_json)
    train_indexes = split_dir / "seen_indexes.json"
    test_indexes = split_dir / "unseen_indexes.json"
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.TRAIN_CLASS_INDEXES") + 1] == str(train_indexes)
    assert overrides[overrides.index("MODEL.SEM_SEG_HEAD.TEST_CLASS_INDEXES") + 1] == str(test_indexes)


def test_trainer_d2_pads_short_taxonomy_when_split_requires_more_classes(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)

    def _fake_train(**kwargs):
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**_kwargs):
        return {"mIoU_all": 10.0, "mIoU_old": 9.0, "mIoU_new": 11.0, "BG-mIoU": 8.0}

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)
    monkeypatch.setattr(
        "covl_seg.engine.open_continual_trainer._resolve_task_class_names",
        lambda _cfg: ["person", "bicycle", "car"],
    )

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2_short_taxonomy",
        engine="d2",
        seed=0,
        method_name="covl",
        clip_finetune="attention",
        task_spec=None,
        num_tasks=1,
        classes_per_task=150,
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

    split_dir = tmp_path / "run_d2_short_taxonomy" / "task_001" / "splits"
    train_names = json.loads((split_dir / "train_class_names.json").read_text(encoding="utf-8"))
    test_names = json.loads((split_dir / "test_class_names.json").read_text(encoding="utf-8"))
    assert len(train_names) >= 150
    assert len(test_names) >= 150
    assert train_names[0] == "person"
    assert train_names[149] == "class_149"


def test_trainer_d2_hands_off_prior_task_checkpoint_via_model_weights(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    class_names = [f"class_{idx}" for idx in range(150)]
    calls = []

    def _fake_train(**kwargs):
        calls.append(kwargs)
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        (out / "model_final.pth").write_text("weights", encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**_kwargs):
        return {"mIoU_all": 10.0, "mIoU_old": 9.0, "mIoU_new": 11.0, "BG-mIoU": 8.0}

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", lambda _cfg: class_names)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2_handoff",
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
    )

    trainer.run()

    assert len(calls) == 2
    second_overrides = calls[1]["extra_overrides"]
    prior_weights = tmp_path / "run_d2_handoff" / "task_001" / "model_final.pth"
    assert second_overrides[second_overrides.index("MODEL.WEIGHTS") + 1] == str(prior_weights)

    state = json.loads((tmp_path / "run_d2_handoff" / "continual_state.json").read_text(encoding="utf-8"))
    assert state["latest_model_path"] == str(tmp_path / "run_d2_handoff" / "task_002" / "model_final.pth")


def test_trainer_d2_enforces_explicit_prior_task_model_weights_handoff(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    class_names = [f"class_{idx}" for idx in range(150)]
    calls = []

    def _fake_train(**kwargs):
        calls.append(kwargs)
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**_kwargs):
        return {"mIoU_all": 10.0, "mIoU_old": 9.0, "mIoU_new": 11.0, "BG-mIoU": 8.0}

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", lambda _cfg: class_names)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2_missing_handoff",
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
    )

    trainer.run()

    assert len(calls) == 2
    second_overrides = calls[1]["extra_overrides"]
    prior_weights = tmp_path / "run_d2_missing_handoff" / "task_001" / "model_final.pth"
    assert second_overrides[second_overrides.index("MODEL.WEIGHTS") + 1] == str(prior_weights)


def test_trainer_d2_persists_expected_latest_model_path_without_existing_file(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    class_names = [f"class_{idx}" for idx in range(150)]

    def _fake_train(**kwargs):
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**_kwargs):
        return {"mIoU_all": 10.0, "mIoU_old": 9.0, "mIoU_new": 11.0, "BG-mIoU": 8.0}

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", lambda _cfg: class_names)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2_expected_latest_path",
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

    state = json.loads((tmp_path / "run_d2_expected_latest_path" / "continual_state.json").read_text(encoding="utf-8"))
    assert state["latest_model_path"] == str(
        tmp_path / "run_d2_expected_latest_path" / "task_001" / "model_final.pth"
    )


def test_trainer_d2_scales_task_train_iters_from_visible_class_budget(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    class_names = [f"class_{idx}" for idx in range(150)]
    calls = []

    def _fake_train(**kwargs):
        calls.append(kwargs)
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        (out / "model_final.pth").write_text("weights", encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**_kwargs):
        return {"mIoU_all": 10.0, "mIoU_old": 9.0, "mIoU_new": 11.0, "BG-mIoU": 8.0}

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", lambda _cfg: class_names)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2_iters",
        engine="d2",
        seed=0,
        method_name="covl",
        clip_finetune="attention",
        task_spec=None,
        num_tasks=2,
        classes_per_task=2,
        task_seed=0,
        n_pre=1,
        n_main=100,
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
        train_iters_mode="on",
        min_iters_per_visible_class=60,
        max_iters_multiplier=10.0,
    )
    trainer.run(max_tasks=2)

    assert len(calls) == 2
    first_overrides = calls[0]["extra_overrides"]
    second_overrides = calls[1]["extra_overrides"]
    first_iters = int(first_overrides[first_overrides.index("SOLVER.MAX_ITER") + 1])
    second_iters = int(second_overrides[second_overrides.index("SOLVER.MAX_ITER") + 1])
    assert first_iters == 120
    assert second_iters == 240


def test_trainer_d2_writes_visible_class_coverage_metrics(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    class_names = [f"class_{idx}" for idx in range(150)]

    def _fake_train(**kwargs):
        out = kwargs["output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text('{"iteration":0,"total_loss":1.0}\n', encoding="utf-8")
        (out / "model_final.pth").write_text("weights", encoding="utf-8")
        return {"num_tasks": 1, "num_phase_records": 1, "last_task": 1}

    def _fake_eval(**kwargs):
        if int(kwargs.get("resume_task", 0)) == 1:
            return {
                "mIoU_all": 10.0,
                "mIoU_old": None,
                "mIoU_new": 11.0,
                "BG-mIoU": 8.0,
                "class_iou_old": {},
                "class_iou_new": {"class_1": 0.5, "class_2": 0.6},
                "class_iou_bg": {},
                "class_iou_all": {"class_1": 0.5, "class_2": 0.6},
            }
        return {
            "mIoU_all": 10.0,
            "mIoU_old": 9.0,
            "mIoU_new": 11.0,
            "BG-mIoU": 8.0,
            "class_iou_old": {"class_1": 0.4},
            "class_iou_new": {"class_3": 0.6, "class_4": 0.7},
            "class_iou_bg": {},
            "class_iou_all": {"class_1": 0.4, "class_3": 0.6, "class_4": 0.7},
        }

    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_train", _fake_train)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer.run_detectron2_eval", _fake_eval)
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", lambda _cfg: class_names)

    trainer = OpenContinualTrainer(
        config_path=str(cfg),
        output_dir=tmp_path / "run_d2_coverage",
        engine="d2",
        seed=0,
        method_name="covl",
        clip_finetune="attention",
        task_spec=None,
        num_tasks=2,
        classes_per_task=2,
        task_seed=0,
        n_pre=1,
        n_main=10,
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
        train_iters_mode="off",
    )
    trainer.run(max_tasks=2)

    metrics_lines = (tmp_path / "run_d2_coverage" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    metrics = [json.loads(line) for line in metrics_lines]
    eval_records = [m for m in metrics if m.get("phase") == "eval"]
    task2_eval = next(m for m in eval_records if int(m.get("task", 0)) == 2)
    assert task2_eval["visible_class_count"] == 4.0
    assert task2_eval["evaluated_visible_class_count"] == 3.0
    assert task2_eval["coverage_visible_ratio"] == 0.75
    assert task2_eval["coverage_new_ratio"] == 1.0
    assert task2_eval["steps_per_visible_class"] == 2.5


def test_resolve_task_class_names_requires_taxonomy(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    empty_project_root = tmp_path / "empty_runtime"
    empty_project_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "covl_seg.engine.open_continual_trainer._d2_project_root",
        lambda: empty_project_root,
    )

    with pytest.raises(ValueError, match="Unable to resolve class names"):
        _resolve_task_class_names(str(cfg))


def test_resolve_task_class_names_uses_d2_runtime_root(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    project_root = tmp_path / "seg_project"
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    expected = ["wall", "floor", "chair"]
    (datasets_dir / "ade150.json").write_text(json.dumps(expected), encoding="utf-8")

    monkeypatch.setenv("COVL_SEG_D2_PROJECT_ROOT", str(project_root))

    assert _resolve_task_class_names(str(cfg)) == expected


def test_resolve_task_class_names_prefers_config_dataset_over_filename(tmp_path, monkeypatch):
    cfg = tmp_path / "copied_from_coco.yaml"
    cfg.write_text(
        """
EXPERIMENT:
  DATASET: ade20k_15
""".strip()
        + "\n",
        encoding="utf-8",
    )
    project_root = tmp_path / "seg_project"
    project_root.mkdir(parents=True, exist_ok=True)
    captured = {}

    def _fake_resolve_class_names(*, project_root, class_json):
        captured["project_root"] = project_root
        captured["class_json"] = class_json
        return ["wall", "floor"]

    monkeypatch.setattr(
        "covl_seg.engine.open_continual_trainer._d2_project_root",
        lambda: project_root,
    )
    monkeypatch.setattr(
        "covl_seg.engine.open_continual_trainer._resolve_class_names",
        _fake_resolve_class_names,
    )

    assert _resolve_task_class_names(str(cfg)) == ["wall", "floor"]
    assert captured["project_root"] == project_root
    assert captured["class_json"].endswith("ade150.json")


def test_resolve_experiment_spec_rejects_unknown_or_ambiguous_dataset(tmp_path):
    unknown_cfg = tmp_path / "copied_config.yaml"
    unknown_cfg.write_text(
        """
EXPERIMENT:
  DATASET: cityscapes
""".strip()
        + "\n",
        encoding="utf-8",
    )

    false_positive_cfg = tmp_path / "false_positive_config.yaml"
    false_positive_cfg.write_text(
        """
EXPERIMENT:
  DATASET: madeup_dataset
""".strip()
        + "\n",
        encoding="utf-8",
    )

    ambiguous_cfg = tmp_path / "ambiguous_config.yaml"
    ambiguous_cfg.write_text(
        """
EXPERIMENT:
  DATASET: coco_ade_combo
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported dataset family"):
        _resolve_experiment_spec(str(unknown_cfg))

    with pytest.raises(ValueError, match="Unsupported dataset family"):
        _resolve_experiment_spec(str(false_positive_cfg))

    with pytest.raises(ValueError, match="Ambiguous dataset family"):
        _resolve_experiment_spec(str(ambiguous_cfg))


def test_resolve_task_class_names_uses_shared_dataset_resolution_for_ade_and_coco(tmp_path, monkeypatch):
    project_root = tmp_path / "seg_project"
    datasets_dir = project_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    ade_names = [f"ade_{idx}" for idx in range(150)]
    coco_names = [f"coco_{idx}" for idx in range(164)]
    (datasets_dir / "ade150.json").write_text(json.dumps(ade_names), encoding="utf-8")
    (datasets_dir / "coco.json").write_text(json.dumps(coco_names), encoding="utf-8")

    ade_cfg = tmp_path / "copied_from_coco.yaml"
    ade_cfg.write_text(
        """
EXPERIMENT:
  DATASET: ade20k_15
""".strip()
        + "\n",
        encoding="utf-8",
    )
    coco_cfg = tmp_path / "copied_from_ade.yaml"
    coco_cfg.write_text(
        """
EXPERIMENT:
  DATASET: coco_stuff_164k
""".strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "covl_seg.engine.open_continual_trainer._d2_project_root",
        lambda: project_root,
    )

    assert _resolve_task_class_names(str(ade_cfg)) == ade_names
    assert _resolve_task_class_names(str(coco_cfg)) == coco_names


def test_trainer_balanced_controller_logs_metrics_and_persists_state(tmp_path, monkeypatch):
    cfg = _write_mock_config(tmp_path)
    class_names = [f"class_{idx}" for idx in range(150)]

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
    monkeypatch.setattr("covl_seg.engine.open_continual_trainer._resolve_task_class_names", lambda _cfg: class_names)

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

    phase_rows = [m for m in metrics if m.get("phase") in {"phase1", "phase2", "phase3", "phase4"}]
    assert phase_rows
    assert all(m.get("proxy_source") == "derived_from_d2_train_metrics" for m in phase_rows)

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
