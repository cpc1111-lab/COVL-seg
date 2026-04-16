# COVL-Seg

Reference implementation workspace for continual open-vocabulary segmentation experiments.

## Quick Start

Run from workspace root `D:\COVL-Seg`.

```bash
# 1) smoke regression run (CPU-safe, 1 task x 5 iters)
python covl_seg/scripts/train_continual.py \
  --config covl_seg/configs/covl_seg_vitb_ade15.yaml \
  --output-dir work_dirs/smoke \
  --seed 0 \
  --smoke
```

```bash
# 2) short local run template (3090)
python covl_seg/scripts/train_continual.py \
  --config covl_seg/configs/covl_seg_vitb_ade15.yaml \
  --output-dir work_dirs/ade15_dev \
  --seed 0 \
  --engine auto \
  --max-tasks 3
```

```bash
# 3) evaluation template (closed-set + optional open-vocab)
python covl_seg/scripts/eval_continual.py \
  --config covl_seg/configs/covl_seg_vitb_ade15.yaml \
  --output-dir work_dirs/ade15_dev \
  --resume-task 3 \
  --engine auto \
  --open-vocab
```

```bash
# 4) generate analysis artifacts from metrics.jsonl
python covl_seg/scripts/make_analysis_figs.py \
  --metrics-jsonl work_dirs/ade15_dev/metrics.jsonl \
  --output-dir work_dirs/ade15_dev/analysis
```

## One-command reproduction recipe

```bash
python covl_seg/scripts/train_continual.py --config covl_seg/configs/covl_seg_vitb_ade15.yaml --output-dir work_dirs/repro_ade15 --seed 0
```

## Notes

- Current scripts are stable entrypoints for argument handling, smoke checks, and artifact generation.
- `train_continual.py` and `eval_continual.py` support `--engine {auto,mock,d2}`.
- `auto` selects Detectron2 when installed; otherwise it falls back to the deterministic mock engine.
- Detectron2 mode runs through `covl_seg/engine/detectron2_runner.py` and writes the same artifact set (`metrics.jsonl`, task checkpoints, `eval_summary.json`).
- Detectron2 mode now builds and optimizes the current `COVLSegModel` composition (backbone + HCIBA + fusion path), while dataset/evaluator are still synthetic scaffolds pending full benchmark wiring.
