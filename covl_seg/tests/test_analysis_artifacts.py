import json

from covl_seg.scripts.make_analysis_figs import generate_analysis_artifacts


def test_generate_analysis_artifacts_writes_proxy_curves(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps({"task": 1, "phase": "phase1", "beta_1_star": 0.1, "beta_2_star": 0.2}),
                json.dumps(
                    {
                        "task": 1,
                        "phase": "phase2",
                        "gamma_clip": 0.7,
                        "ctr_loss": 0.03,
                        "alpha_floor": 0.05,
                        "delta_new": 0.12,
                        "delta_old": 0.08,
                        "delta_all": 0.1,
                        "ov_min_delta": 0.08,
                    }
                ),
                json.dumps(
                    {
                        "task": 1,
                        "phase": "phase3",
                        "omega_tau_t": 0.4,
                        "tau_pred": 0.06,
                        "ov_guard_triggered": True,
                        "ov_guard_state": "active",
                    }
                ),
                json.dumps({"task": 2, "phase": "phase1", "beta_1_star": 0.11, "beta_2_star": 0.21}),
                json.dumps(
                    {
                        "task": 2,
                        "phase": "phase2",
                        "gamma_clip": 0.71,
                        "ctr_loss": 0.031,
                        "alpha_floor": 0.06,
                        "delta_new": 0.13,
                        "delta_old": 0.09,
                        "delta_all": 0.11,
                        "ov_min_delta": 0.09,
                    }
                ),
                json.dumps(
                    {
                        "task": 2,
                        "phase": "phase3",
                        "omega_tau_t": 0.41,
                        "tau_pred": 0.061,
                        "ov_guard_triggered": False,
                        "ov_guard_state": "inactive",
                    }
                ),
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
    assert "alpha_floor" in curves
    assert "delta_new" in curves
    assert "delta_old" in curves
    assert "delta_all" in curves
    assert "ov_min_delta" in curves

    task_summary = json.loads((out / "analysis_task_summary.json").read_text(encoding="utf-8"))
    assert "1" in task_summary
    assert "2" in task_summary
    assert task_summary["1"]["beta_1_star"] == 0.1
    assert task_summary["2"]["omega_tau_t"] == 0.41
    assert task_summary["1"]["alpha_floor"] == 0.05
    assert task_summary["2"]["delta_new"] == 0.13
    assert task_summary["2"]["delta_old"] == 0.09
    assert task_summary["2"]["delta_all"] == 0.11
    assert task_summary["2"]["ov_min_delta"] == 0.09
    assert task_summary["1"]["ov_guard_triggered"] is True
    assert task_summary["2"]["ov_guard_state"] == "inactive"


def test_generate_analysis_artifacts_falls_back_to_eval_metrics_when_eval_summary_missing(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": 1,
                        "phase": "eval",
                        "mIoU_all": 55.0,
                        "class_iou_all": {"cls_a": 40.0, "cls_b": 50.0},
                        "class_iou_old": {"cls_a": 40.0},
                        "class_iou_new": {"cls_b": 50.0},
                        "class_iou_bg": {},
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "analysis"
    payload = generate_analysis_artifacts(metrics_jsonl=metrics, output_dir=out, run_dir=tmp_path)
    assert payload["num_records"] == 1

    group_rows = json.loads((out / "analysis_group_trends.json").read_text(encoding="utf-8"))
    assert len(group_rows) == 1
    assert group_rows[0]["count_old"] == 1
    assert group_rows[0]["count_new"] == 1

    class_trends = json.loads((out / "analysis_class_iou_trends.json").read_text(encoding="utf-8"))
    assert "cls_a" in class_trends
    assert "cls_b" in class_trends
