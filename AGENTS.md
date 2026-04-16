# AGENTS.md

High-signal workspace notes for OpenCode sessions.

## Scope and layout

- Workspace has two active codebases:
  - `covl_seg/`: current continual open-vocabulary segmentation workspace with real pytest coverage.
  - `CAT-Seg/`: upstream CAT-Seg training/eval code (Detectron2-style scripts).
- Run commands from workspace root (`/home/e222/cpc/COVL-Seg`) unless a command explicitly expects `CAT-Seg/` paths.

## Verified entrypoints (covl_seg)

- Use module execution from repo root, not file execution.
  - Works: `python -m covl_seg.scripts.train_continual ...`
  - Fails from root: `python covl_seg/scripts/train_continual.py ...` (`ModuleNotFoundError: covl_seg`).
- Main scripts:
  - Train: `python -m covl_seg.scripts.train_continual --config covl_seg/configs/covl_seg_vitb_ade15.yaml --output-dir work_dirs/dev --seed 0`
  - Eval: `python -m covl_seg.scripts.eval_continual --config covl_seg/configs/covl_seg_vitb_ade15.yaml --output-dir work_dirs/dev --resume-task 1 --open-vocab`
  - Analysis: `python -m covl_seg.scripts.make_analysis_figs --metrics-jsonl work_dirs/dev/metrics.jsonl --output-dir work_dirs/dev/analysis`
- `--engine auto` is the default; it picks Detectron2 if installed, else deterministic mock backend.

## Fast verification commands

- Full test suite: `python -m pytest covl_seg/tests -q` (currently 33 tests, ~1s).
- Single test file: `python -m pytest covl_seg/tests/test_train_eval_integration.py -q`
- Single test case: `python -m pytest covl_seg/tests/test_train_eval_integration.py::test_eval_once_reads_train_artifacts_and_writes_summary -q`
- Smoke run (no Detectron2 required): `python -m covl_seg.scripts.train_continual --config covl_seg/configs/covl_seg_vitb_ade15.yaml --output-dir work_dirs/smoke --seed 0 --smoke`

## Artifacts and flow quirks

- Train script always writes `run_config.json`; eval writes `eval_config.json` in `--output-dir`.
- Mock/D2 train writes `metrics.jsonl` and `checkpoint_task_XXX.json`; eval writes `eval_summary.json`.
- `covl_seg/scripts/prepare_data.sh` is a placeholder (prints TODO, does not prepare datasets yet).

## CAT-Seg notes (legacy but used)

- `CAT-Seg/run.sh` trains then immediately calls `CAT-Seg/eval.sh`.
- `CAT-Seg/eval.sh` runs multiple dataset evals and defaults `MODEL.WEIGHTS` to `$output/model_final.pth` unless overridden via OPTS.
- Canonical commands:
  - `sh CAT-Seg/run.sh CAT-Seg/configs/vitb_384.yaml 4 output/`
  - `sh CAT-Seg/eval.sh CAT-Seg/configs/vitl_336.yaml 4 output/ MODEL.WEIGHTS path/to/weights.pth`

## Instruction files check

- No `.cursor/rules/`, `.cursorrules`, `.github/copilot-instructions.md`, or `opencode.json` found in this workspace.
