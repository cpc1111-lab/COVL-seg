from pathlib import Path


def test_smoke_run_writes_metrics_and_is_finite(tmp_path: Path):
    from covl_seg.scripts.train_continual import run_smoke_once

    out_dir = tmp_path / "smoke"
    payload = run_smoke_once(output_dir=out_dir, seed=0)

    assert payload["num_tasks"] == 1
    assert payload["num_iters"] == 5
    assert payload["has_nan"] is False
    assert (out_dir / "smoke_metrics.json").exists()
