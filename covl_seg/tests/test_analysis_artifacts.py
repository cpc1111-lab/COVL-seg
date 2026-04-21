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
    run_dir = tmp_path / "run"
    task1 = run_dir / "task_001"
    task2 = run_dir / "task_002"
    task1.mkdir(parents=True, exist_ok=True)
    task2.mkdir(parents=True, exist_ok=True)
    (task1 / "eval_summary.json").write_text(
        json.dumps(
            {
                "class_iou_all": {"person": 10.0, "car": 20.0},
                "class_iou_old": {"person": 10.0},
                "class_iou_new": {"car": 20.0},
                "class_iou_bg": {},
            }
        ),
        encoding="utf-8",
    )
    (task2 / "eval_summary.json").write_text(
        json.dumps(
            {
                "class_iou_all": {"person": 30.0, "car": 40.0},
                "class_iou_old": {"person": 30.0, "car": 40.0},
                "class_iou_new": {},
                "class_iou_bg": {},
            }
        ),
        encoding="utf-8",
    )

    payload = generate_analysis_artifacts(metrics_jsonl=metrics, output_dir=out, run_dir=run_dir)
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

    class_trends = json.loads((out / "analysis_class_iou_trends.json").read_text(encoding="utf-8"))
    assert "car" in class_trends
    assert class_trends["car"][0]["group"] == "new"
    assert class_trends["car"][1]["group"] == "old"

    assert (out / "fig_group_miou_trends.png").exists()
    assert (out / "fig_class_trends" / "car.png").exists()
