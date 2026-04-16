from pathlib import Path

import torch


def test_open_vocab_evaluator_reports_dataset_metric():
    from covl_seg.engine.open_vocab_eval import OpenVocabEvaluator

    evaluator = OpenVocabEvaluator(dataset_aliases={"pc59": "pascal_context_59"})

    logits = torch.randn(1, 3, 8, 8)
    targets = torch.randint(0, 3, (1, 8, 8))
    metrics = evaluator.evaluate_dataset(dataset_key="pc59", logits=logits, targets=targets)

    assert "ov_mIoU_pc59" in metrics
    assert isinstance(metrics["ov_mIoU_pc59"], float)


def test_make_analysis_figs_generates_outputs(tmp_path: Path):
    from covl_seg.scripts.make_analysis_figs import generate_analysis_artifacts

    metrics_jsonl = tmp_path / "metrics.jsonl"
    metrics_jsonl.write_text(
        "\n".join(
            [
                '{"task": 1, "beta_1": 0.2, "beta_2": 0.3, "tau_pred": 0.07}',
                '{"task": 2, "beta_1": 0.4, "beta_2": 0.5, "tau_pred": 0.06}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "figs"
    result = generate_analysis_artifacts(metrics_jsonl=metrics_jsonl, output_dir=out_dir)

    assert (out_dir / "analysis_summary.csv").exists()
    assert (out_dir / "analysis_summary.json").exists()
    assert result["num_records"] == 2
