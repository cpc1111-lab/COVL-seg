import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_full_fixture_8_pngs_generated(tmp_path):
    from covl_seg.engine.report_generator import generate_report

    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"task": 1, "phase": "phase1", "I_exc_C": 0.5, "I_exc_S": 0.6}),
                json.dumps({"task": 1, "phase": "phase2", "ctr_loss": -0.03, "gamma_clip": 0.7}),
                json.dumps(
                    {
                        "task": 1,
                        "phase": "phase3",
                        "alpha_star": 0.45,
                        "tau_pred": 0.07,
                        "fisher_energy": 0.8,
                        "omega_tau_t": 0.3,
                    }
                ),
                json.dumps({"task": 1, "phase": "phase4", "replay_priority_total": 5.0, "replay_selected": 10}),
                json.dumps(
                    {
                        "task": 1,
                        "phase": "eval",
                        "mIoU_all": 70.0,
                        "mIoU_old": 75.0,
                        "mIoU_new": 65.0,
                        "BG-mIoU": 60.0,
                    }
                ),
                json.dumps({"task": 2, "phase": "phase1", "I_exc_C": 0.52, "I_exc_S": 0.65}),
                json.dumps({"task": 2, "phase": "phase2", "ctr_loss": -0.032, "gamma_clip": 0.72}),
                json.dumps(
                    {
                        "task": 2,
                        "phase": "phase3",
                        "alpha_star": 0.46,
                        "tau_pred": 0.068,
                        "fisher_energy": 0.85,
                        "omega_tau_t": 0.32,
                    }
                ),
                json.dumps({"task": 2, "phase": "phase4", "replay_priority_total": 6.0, "replay_selected": 12}),
                json.dumps(
                    {
                        "task": 2,
                        "phase": "eval",
                        "mIoU_all": 72.0,
                        "mIoU_old": 70.0,
                        "mIoU_new": 74.0,
                        "BG-mIoU": 62.0,
                    }
                ),
                json.dumps({"task": 3, "phase": "phase1", "I_exc_C": 0.55, "I_exc_S": 0.7}),
                json.dumps({"task": 3, "phase": "phase2", "ctr_loss": -0.035, "gamma_clip": 0.75}),
                json.dumps(
                    {
                        "task": 3,
                        "phase": "phase3",
                        "alpha_star": 0.48,
                        "tau_pred": 0.065,
                        "fisher_energy": 0.9,
                        "omega_tau_t": 0.35,
                    }
                ),
                json.dumps({"task": 3, "phase": "phase4", "replay_priority_total": 7.0, "replay_selected": 14}),
                json.dumps(
                    {
                        "task": 3,
                        "phase": "eval",
                        "mIoU_all": 75.0,
                        "mIoU_old": 68.0,
                        "mIoU_new": 80.0,
                        "BG-mIoU": 65.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "analysis"
    result = generate_report(run_dir=tmp_path, output_dir=out)

    expected_figs = [
        "fig_perf_miou_curves",
        "fig_perf_forgetting",
        "fig_perf_stability_plasticity",
        "fig_theory_alpha_tau",
        "fig_theory_iexc",
        "fig_theory_spectral",
        "fig_bg_ctr",
        "fig_sacr_replay",
    ]
    for fig_name in expected_figs:
        assert fig_name in result, f"{fig_name} not in result"
        path = result[fig_name]
        assert path.exists(), f"{fig_name} path does not exist"
        assert path.stat().st_size > 0, f"{fig_name} is empty"

    assert out.exists(), "output_dir was not created"


def test_missing_fields_only_matching_figures(tmp_path):
    from covl_seg.engine.report_generator import generate_report

    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"task": 1, "phase": "phase1", "I_exc_C": 0.5, "I_exc_S": 0.6}),
                json.dumps({"task": 2, "phase": "phase1", "I_exc_C": 0.52, "I_exc_S": 0.65}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = generate_report(run_dir=tmp_path)

    assert "fig_theory_iexc" in result
    assert "fig_perf_miou_curves" not in result
    assert "fig_perf_forgetting" not in result
    assert "fig_perf_stability_plasticity" not in result
    assert "fig_theory_alpha_tau" not in result
    assert "fig_theory_spectral" not in result
    assert "fig_bg_ctr" not in result
    assert "fig_sacr_replay" not in result


def test_missing_metrics_jsonl_returns_empty(tmp_path):
    from covl_seg.engine.report_generator import generate_report

    result = generate_report(run_dir=tmp_path)

    assert result == {}


def test_completed_zero_generate_report_not_called(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("ADE20K_15\n", encoding="utf-8")
    from covl_seg.engine import open_continual_trainer as trainer_module
    trainer_class = trainer_module.OpenContinualTrainer
    trainer = trainer_class(
        config_path=str(config_path),
        output_dir=tmp_path,
        engine="auto",
        seed=0,
        method_name="none",
        clip_finetune="none",
        task_spec=None,
        num_tasks=1,
        classes_per_task=10,
        task_seed=0,
        n_pre=1,
        n_main=1,
        eps_f=0.05,
        t_mem="1",
        mix_ratio=[1],
        m_max_total=100,
        m_max_per_class=10,
        ewc_lambda=0.0,
        ewc_topk=1,
        ewc_iters=1,
        enable_ciba=False,
        enable_ctr=False,
        enable_spectral_ogp=False,
        enable_sacr=False,
        resume_task=1,
    )
    result = trainer.run(max_tasks=0)
    analysis_dir = tmp_path / "analysis"
    assert not analysis_dir.exists()


def test_make_analysis_figs_return_value_contains_figures_generated(tmp_path):
    from covl_seg.engine.report_generator import generate_report
    from covl_seg.scripts.make_analysis_figs import generate_analysis_artifacts

    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"task": 1, "phase": "phase1", "I_exc_C": 0.5, "I_exc_S": 0.6}),
                json.dumps({"task": 1, "phase": "phase2", "ctr_loss": -0.03, "gamma_clip": 0.7}),
                json.dumps(
                    {
                        "task": 1,
                        "phase": "phase3",
                        "alpha_star": 0.45,
                        "tau_pred": 0.07,
                        "fisher_energy": 0.8,
                        "omega_tau_t": 0.3,
                    }
                ),
                json.dumps(
                    {
                        "task": 1,
                        "phase": "eval",
                        "mIoU_all": 70.0,
                        "mIoU_old": 75.0,
                        "mIoU_new": 65.0,
                        "BG-mIoU": 60.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "analysis"
    payload = generate_analysis_artifacts(metrics_jsonl=metrics, output_dir=out, run_dir=tmp_path)

    assert "figures_generated" in payload

    summary_json = out / "analysis_summary.json"
    summary_data = json.loads(summary_json.read_text(encoding="utf-8"))
    assert "figures_generated" not in summary_data