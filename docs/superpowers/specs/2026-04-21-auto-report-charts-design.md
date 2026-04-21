# Auto Report Charts Design
**Date:** 2026-04-21
**Topic:** Automatic post-training statistical chart generation

## Overview

After each training run, COVL-Seg automatically generates a set of PNG charts covering
segmentation performance and continual open-vocabulary learning (COVL) theory quantities.
Charts are also producible on demand via the existing `make_analysis_figs.py` CLI.

## Architecture

```
covl_seg/
├── engine/
│   ├── report_generator.py          # NEW — all chart rendering functions
│   └── open_continual_trainer.py    # MODIFIED — auto-call at end of run()
├── scripts/
│   └── make_analysis_figs.py        # MODIFIED — delegates to report_generator
└── tests/
    └── test_report_generator.py     # NEW — per-group chart tests
```

**Data flow:**

```
metrics.jsonl
task_*/eval_summary.json  →  generate_report(run_dir)
                                      │
                           {run_dir}/analysis/
                               fig_perf_*.png      # Group 1: performance
                               fig_theory_*.png    # Group 2: theory quantities
                               fig_bg_*.png        # Group 3: background entropy
                               fig_sacr_*.png      # Group 4: SACR replay
```

## Public API

```python
# covl_seg/engine/report_generator.py
def generate_report(
    run_dir: Path,
    output_dir: Path | None = None,  # default: run_dir / "analysis"
) -> Dict[str, Path]:                # {"fig_perf_miou_curves": Path(...), ...}
```

Returns only the figures that were successfully generated. Skipped figures
(missing data) are absent from the dict. No exceptions propagate to callers.

## Chart Inventory (8 figures)

### Group 1 — Segmentation Performance (`fig_perf_*.png`)

| File | Content | Data source |
|------|---------|-------------|
| `fig_perf_miou_curves.png` | mIoU_all / old / new / BG-mIoU line chart across tasks | `metrics.jsonl` eval records |
| `fig_perf_forgetting.png` | Per-task backward transfer: bar = `mIoU_new[τ] − mIoU_old[τ]` for each task τ>1, line = cumulative BWT approximation. Falls back to `delta_old` from `balanced_ctrl` records when available (balanced profile only). Skipped entirely if fewer than 2 eval records exist. | eval records (primary); `balanced_ctrl` records (supplemental) |
| `fig_perf_stability_plasticity.png` | mIoU_old vs mIoU_new scatter, one point per task, diagonal reference line | eval records |

### Group 2 — SIGF Theory Quantities (`fig_theory_*.png`)

| File | Content | Data source |
|------|---------|-------------|
| `fig_theory_alpha_tau.png` | α*(t) left-axis line + τ_pred right-axis line (dual y-axis) | phase3 records |
| `fig_theory_iexc.png` | I_exc^C and I_exc^S dual lines with shaded gap representing Δ_S^C, computed at render time as `max(0, I_exc_S − I_exc_C)` from the two stored fields (no separate Δ_S^C field exists in metrics.jsonl) | phase1 records |
| `fig_theory_spectral.png` | Fisher energy bar chart (left axis) + ω_{τ,t} line (right axis) | phase3 records |

### Group 3 — Background Entropy Control (`fig_bg_*.png`)

| File | Content | Data source |
|------|---------|-------------|
| `fig_bg_ctr.png` | |CTR loss| line (left axis) + γ_clip line (right axis) | phase2 records |

### Group 4 — SACR Replay (`fig_sacr_*.png`)

| File | Content | Data source |
|------|---------|-------------|
| `fig_sacr_replay.png` | replay_priority_total bar chart (left axis) + replay_selected line (right axis) | phase4 records |

## Integration

### Auto-trigger (`OpenContinualTrainer.run()`)

Inserted before `return`, guarded by `completed > 0` (prevents running when all tasks were
skipped via `resume_task`, at which point `metrics.jsonl` may not exist):

```python
if completed > 0:
    try:
        from covl_seg.engine.report_generator import generate_report
        generated = generate_report(run_dir=self.output_dir)
        _log.info("[report] generated %d figures → %s", len(generated), self.output_dir / "analysis")
    except Exception as exc:
        _log.warning("[report] chart generation skipped: %s", exc)
```

Where `_log = logging.getLogger(__name__)` is module-level.
`generate_report` handles a missing `metrics.jsonl` by returning `{}` immediately.
Chart failure never interrupts training result delivery.

### Manual trigger (`make_analysis_figs.py`)

`generate_analysis_artifacts()` appends after existing data processing:

```python
from covl_seg.engine.report_generator import generate_report
generated = generate_report(run_dir=actual_run_dir, output_dir=output_dir / "analysis")
summary_payload["figures_generated"] = list(generated.keys())
```

`figures_generated` is added only to the **in-memory return value** of `generate_analysis_artifacts()`.
It is **not** written to `analysis_summary.json` on disk (that file's schema is unchanged).
CLI arguments unchanged.

## Missing Data Handling

- Any field absent or all-NaN across all tasks → that figure is skipped silently
- Skipped figures: absent from return dict; logged via `_log.warning("[report] skipped %s: no data", fig_name)` — consistent with the `logging` approach used in the auto-trigger block, never `print()`
- `fig_perf_forgetting` and `fig_perf_stability_plasticity` both require at least 2 eval records; skipped if fewer exist
- `output_dir` is created (with `mkdir(parents=True, exist_ok=True)`) only on the first successful figure write, not on entry — so a run that produces no figures leaves no empty directory

## Tests (`test_report_generator.py`)

| Test | Method |
|------|--------|
| Full fixture → 8 PNGs generated, non-empty, `analysis/` dir created | Write complete fixture metrics.jsonl + eval_summary.json stubs; assert all 8 PNG paths exist, size > 0, and `output_dir` was created |
| Missing fields → only matching figures generated, no exception | Fixture with only phase1 records; assert only `fig_theory_iexc` produced, no other paths present, no exception raised |
| Missing `metrics.jsonl` → returns `{}`, no exception | Call `generate_report` on empty dir; assert return is `{}` |
| `completed=0` → `generate_report` not called | Mock `covl_seg.engine.open_continual_trainer.generate_report` (the name bound by the deferred import inside `run()`); assert mock not called when zero tasks execute |
| `make_analysis_figs` return value contains `figures_generated` | Call `generate_analysis_artifacts()` directly; assert `"figures_generated"` key in return dict but NOT in written `analysis_summary.json` |

Existing `test_analysis_artifacts.py` is not modified.
