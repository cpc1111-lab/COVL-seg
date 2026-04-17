import json

from covl_seg.scripts.make_analysis_figs import generate_analysis_artifacts


def test_generate_analysis_artifacts_writes_proxy_curves(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"task": 1, "phase": "phase1", "beta_1_star": 0.1, "beta_2_star": 0.2}),
                json.dumps({"task": 1, "phase": "phase2", "gamma_clip": 0.7, "ctr_loss": 0.03}),
                json.dumps({"task": 1, "phase": "phase3", "omega_tau_t": 0.4, "tau_pred": 0.06}),
                json.dumps({"task": 2, "phase": "phase1", "beta_1_star": 0.11, "beta_2_star": 0.21}),
                json.dumps({"task": 2, "phase": "phase2", "gamma_clip": 0.71, "ctr_loss": 0.031}),
                json.dumps({"task": 2, "phase": "phase3", "omega_tau_t": 0.41, "tau_pred": 0.061}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "analysis"
    payload = generate_analysis_artifacts(metrics_jsonl=metrics, output_dir=out)
    assert payload["num_records"] == 6
    curves = json.loads((out / "analysis_curves.json").read_text(encoding="utf-8"))
    assert "beta_1_star" in curves
    assert "gamma_clip" in curves
    assert "omega_tau_t" in curves

    task_summary = json.loads((out / "analysis_task_summary.json").read_text(encoding="utf-8"))
    assert "1" in task_summary
    assert "2" in task_summary
    assert task_summary["1"]["beta_1_star"] == 0.1
    assert task_summary["2"]["omega_tau_t"] == 0.41
