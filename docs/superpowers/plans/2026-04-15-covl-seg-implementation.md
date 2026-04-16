# COVL-Seg Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the COVL-Seg method and evaluation pipeline on top of Detectron2/CAT-Seg, with reproducible continual-learning experiments and theory-validation outputs.

**Architecture:** Keep `CAT-Seg/` unchanged as upstream. Implement all new logic in sibling package `covl_seg/`, with a 4-phase trainer loop (HCIBA pre-alignment, joint training with spectral projection, Fisher subspace extraction/fusion parameter estimation, SACR replay updates). Use Detectron2 hooks/evaluator plumbing and file-based experiment artifacts.

**Tech Stack:** Python 3.8, PyTorch, Detectron2, CAT-Seg CLIP wrappers, pytest, JSON/CSV logging.

---

## Chunk 1: Foundation (M0-M1)

### Task 1: Repository skeleton and package wiring (M0)

**Files:**
- Create: `covl_seg/__init__.py`
- Create: `covl_seg/model/__init__.py`
- Create: `covl_seg/losses/__init__.py`
- Create: `covl_seg/continual/__init__.py`
- Create: `covl_seg/data/__init__.py`
- Create: `covl_seg/engine/__init__.py`
- Create: `covl_seg/baselines/__init__.py`
- Create: `covl_seg/configs/.gitkeep`
- Create: `covl_seg/scripts/.gitkeep`
- Create: `covl_seg/tests/test_import_smoke.py`

- [ ] **Step 1: Create package directories and module init files**
- [ ] **Step 2: Add minimal exports in each `__init__.py` (no business logic yet)**
- [ ] **Step 3: Write import smoke test**

```python
def test_import_covl_seg():
    import covl_seg  # noqa: F401
```

- [ ] **Step 4: Run smoke test**

Run: `pytest covl_seg/tests/test_import_smoke.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add covl_seg
git commit -m "chore: scaffold covl_seg package and import smoke test"
```

### Task 2: Model module scaffolds and typed interfaces (M1)

**Files:**
- Create: `covl_seg/model/covl_seg_model.py`
- Create: `covl_seg/model/continual_backbone.py`
- Create: `covl_seg/model/hciba_head.py`
- Create: `covl_seg/model/boundary_detect.py`
- Create: `covl_seg/model/fusion.py`
- Test: `covl_seg/tests/test_model_shapes.py`

- [ ] **Step 1: Write failing shape test for end-to-end model forward on dummy input**

```python
def test_covl_seg_model_forward_shape():
    # instantiate minimal model with mock CLIP adapter and text embeddings
    # assert output logits shape [B, C, H, W]
    ...
```

- [ ] **Step 2: Run test to confirm failure (missing modules/classes)**

Run: `pytest covl_seg/tests/test_model_shapes.py::test_covl_seg_model_forward_shape -q`
Expected: FAIL with ImportError/AttributeError

- [ ] **Step 3: Implement minimal working versions of model components**
  - `ContinualBackbone`: token-grid reshape + upsample contract
  - `BoundaryDetector`: CLIP attention-gradient API + Sobel fallback switch
  - `HCIBAHead`: `phi_bnd` and `phi_sem` linear projections
  - `FusionHead`: PoE-style logit fusion + temperature scaling
  - `COVLSegModel`: compose modules and return train/infer dicts

- [ ] **Step 4: Re-run shape test**

Run: `pytest covl_seg/tests/test_model_shapes.py -q`
Expected: PASS

- [ ] **Step 5: Add gradient sanity test and run**

Run: `pytest covl_seg/tests/test_model_shapes.py::test_backward_no_nan -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add covl_seg/model covl_seg/tests/test_model_shapes.py
git commit -m "feat: add covl_seg model scaffold with shape-safe forward/backward"
```

## Chunk 2: Losses and Continual Core (M2-M3)

### Task 3: Loss implementations with numeric unit tests (M2)

**Files:**
- Create: `covl_seg/losses/mine.py`
- Create: `covl_seg/losses/ciba.py`
- Create: `covl_seg/losses/ctr.py`
- Create: `covl_seg/losses/segmentation.py`
- Test: `covl_seg/tests/test_losses_numeric.py`

- [ ] **Step 1: Write failing tests for each loss contract**
  - MI estimator returns finite scalar
  - beta-star estimator finite and clipped
  - CTR loss increases penalty on confusing background classes
  - segmentation CE path works on masked labels

- [ ] **Step 2: Run tests and verify failures**

Run: `pytest covl_seg/tests/test_losses_numeric.py -q`
Expected: FAIL

- [ ] **Step 3: Implement minimal passing losses with clear function boundaries**
- [ ] **Step 4: Add Gaussian-pair MI sanity test (monotonic trend)**
- [ ] **Step 5: Run full loss tests**

Run: `pytest covl_seg/tests/test_losses_numeric.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add covl_seg/losses covl_seg/tests/test_losses_numeric.py
git commit -m "feat: implement COVL losses with numerical sanity tests"
```

### Task 4: Fisher subspace and spectral projection (M3-A)

**Files:**
- Create: `covl_seg/continual/fisher.py`
- Create: `covl_seg/continual/spectral_ogp.py`
- Test: `covl_seg/tests/test_spectral_projection.py`
- Test: `covl_seg/tests/test_fisher_power_iteration.py`

- [ ] **Step 1: Write failing tests for projection orthogonality and eigenspace recovery**
- [ ] **Step 2: Run failing tests**

Run: `pytest covl_seg/tests/test_spectral_projection.py covl_seg/tests/test_fisher_power_iteration.py -q`
Expected: FAIL

- [ ] **Step 3: Implement implicit Fisher matvec and top-k power iteration + deflation**
- [ ] **Step 4: Implement hard projection `g <- g - V(V^T g)` per layer block**
- [ ] **Step 5: Re-run tests and verify tolerance-based assertions**

Run: `pytest covl_seg/tests/test_spectral_projection.py covl_seg/tests/test_fisher_power_iteration.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add covl_seg/continual covl_seg/tests/test_spectral_projection.py covl_seg/tests/test_fisher_power_iteration.py
git commit -m "feat: add Fisher subspace extraction and spectral OGP projection"
```

### Task 5: SACR replay buffer and priority policy (M3-B)

**Files:**
- Create: `covl_seg/continual/replay_buffer.py`
- Test: `covl_seg/tests/test_replay_buffer.py`

- [ ] **Step 1: Write failing tests for add/evict ordering and max-per-class constraints**
- [ ] **Step 2: Implement replay item schema and priority update policy**
- [ ] **Step 3: Implement deterministic eviction and serialization helpers**
- [ ] **Step 4: Run tests**

Run: `pytest covl_seg/tests/test_replay_buffer.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add covl_seg/continual/replay_buffer.py covl_seg/tests/test_replay_buffer.py
git commit -m "feat: implement SACR replay buffer with priority-based eviction"
```

## Chunk 3: Data, Engine, and Metrics (M4-M5)

### Task 6: Continual dataset splits and mixed loader (M4)

**Files:**
- Create: `covl_seg/data/ade20k_15.py`
- Create: `covl_seg/data/ade20k_100.py`
- Create: `covl_seg/data/coco_stuff_164_10.py`
- Create: `covl_seg/data/continual_loader.py`
- Create: `covl_seg/data/splits/ade20k_15.json`
- Create: `covl_seg/data/splits/ade20k_100.json`
- Create: `covl_seg/data/splits/coco_stuff_164_10.json`
- Test: `covl_seg/tests/test_continual_splits.py`

- [ ] **Step 1: Write failing split integrity tests**
  - union coverage equals full class set
  - no duplicate class assignment
  - per-task supervision only on `C_t`

- [ ] **Step 2: Implement split loaders and frozen ordering readers**
- [ ] **Step 3: Implement `ContinualDataLoader` with configurable `D_t:M` ratio**
- [ ] **Step 4: Run split and loader tests**

Run: `pytest covl_seg/tests/test_continual_splits.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add covl_seg/data covl_seg/tests/test_continual_splits.py
git commit -m "feat: add continual split definitions and mixed-task dataloader"
```

### Task 7: Four-phase trainer + hooks + closed-set evaluator (M5)

**Files:**
- Create: `covl_seg/engine/trainer.py`
- Create: `covl_seg/engine/hooks.py`
- Create: `covl_seg/engine/evaluator.py`
- Test: `covl_seg/tests/test_trainer_phase_switch.py`
- Test: `covl_seg/tests/test_metrics_logging.py`

- [ ] **Step 1: Write failing tests for phase freeze/unfreeze policy**
- [ ] **Step 2: Implement trainer outer task loop (`for t in range(T)`)**
- [ ] **Step 3: Implement Phase 1/2/3/4 methods and checkpoints per task**
- [ ] **Step 4: Implement hooks to write `metrics.jsonl` proxies**
- [ ] **Step 5: Implement evaluator outputs (`mIoU_all/old/new`, `BG-mIoU`, `BWT` inputs)**
- [ ] **Step 6: Run trainer-engine tests**

Run: `pytest covl_seg/tests/test_trainer_phase_switch.py covl_seg/tests/test_metrics_logging.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add covl_seg/engine covl_seg/tests/test_trainer_phase_switch.py covl_seg/tests/test_metrics_logging.py
git commit -m "feat: implement four-phase continual trainer with metrics hooks"
```

## Chunk 4: Baselines, Open-Vocab, and Reproducibility (M6-M10)

### Task 8: Experiment configs and runnable scripts (M6 support)

**Files:**
- Create: `covl_seg/configs/covl_seg_vitb_ade15.yaml`
- Create: `covl_seg/configs/covl_seg_vitb_ade100.yaml`
- Create: `covl_seg/configs/covl_seg_vitb_coco.yaml`
- Create: `covl_seg/configs/covl_seg_vitl_ade15.yaml`
- Create: `covl_seg/scripts/train_continual.py`
- Create: `covl_seg/scripts/eval_continual.py`
- Create: `covl_seg/scripts/prepare_data.sh`

- [ ] **Step 1: Add minimal config schema and defaults from spec §12**
- [ ] **Step 2: Implement CLI scripts for train/eval with resume-from-task**
- [ ] **Step 3: Add one-task smoke command in script help text**
- [ ] **Step 4: Run syntax checks**

Run: `python -m py_compile covl_seg/scripts/train_continual.py covl_seg/scripts/eval_continual.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add covl_seg/configs covl_seg/scripts
git commit -m "feat: add COVL experiment configs and train/eval entry scripts"
```

### Task 9: Baseline wrappers (M7)

**Files:**
- Create: `covl_seg/baselines/ft.py`
- Create: `covl_seg/baselines/oracle.py`
- Create: `covl_seg/baselines/ewc.py`
- Create: `covl_seg/baselines/mib.py`
- Create: `covl_seg/baselines/plop.py`
- Create: `covl_seg/baselines/zscl.py`
- Test: `covl_seg/tests/test_baseline_switches.py`

- [ ] **Step 1: Write failing tests asserting each baseline toggles expected modules/loss terms**
- [ ] **Step 2: Implement baseline mode registry and wrappers**
- [ ] **Step 3: Run baseline tests**

Run: `pytest covl_seg/tests/test_baseline_switches.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add covl_seg/baselines covl_seg/tests/test_baseline_switches.py
git commit -m "feat: add baseline mode wrappers for continual experiments"
```

### Task 10: Open-vocabulary evaluator and analysis figure generator (M8 + M10 partial)

**Files:**
- Create: `covl_seg/engine/open_vocab_eval.py`
- Create: `covl_seg/scripts/make_analysis_figs.py`
- Test: `covl_seg/tests/test_open_vocab_eval.py`

- [ ] **Step 1: Write failing tests for class-name encoding and dataset metric reporting**
- [ ] **Step 2: Implement open-vocab evaluator for PC-59/PC-459/VOC-20**
- [ ] **Step 3: Implement figure script for proxies (`Omega`, `beta*`, `tau_pred`, etc.)**
- [ ] **Step 4: Run tests**

Run: `pytest covl_seg/tests/test_open_vocab_eval.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add covl_seg/engine/open_vocab_eval.py covl_seg/scripts/make_analysis_figs.py covl_seg/tests/test_open_vocab_eval.py
git commit -m "feat: add open-vocabulary evaluation and theory-proxy plotting"
```

### Task 11: End-to-end smoke, docs, and reproducibility recipe (M6 + M9 + M10 closeout)

**Files:**
- Modify: `CAT-Seg/README.md` (or create `covl_seg/README.md` if preferred)
- Create: `covl_seg/tests/test_end_to_end_smoke.py`
- Modify: `covl_seg/scripts/train_continual.py` (if smoke mode flag needed)

- [ ] **Step 1: Add failing end-to-end smoke test scaffold (1 task x 5 iters CPU mode)**
- [ ] **Step 2: Implement smoke mode and deterministic seed path**
- [ ] **Step 3: Run smoke test**

Run: `pytest covl_seg/tests/test_end_to_end_smoke.py -q`
Expected: PASS within time budget

- [ ] **Step 4: Add reproducibility commands and expected outputs to README**
  - local smoke command
  - 3090 short run command
  - 4090 production command template
  - analysis table/figure generation command

- [ ] **Step 5: Run broad test suite**

Run: `pytest covl_seg/tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add covl_seg/tests covl_seg/scripts CAT-Seg/README.md
git commit -m "docs: add reproduction recipe and validate end-to-end smoke flow"
```

## Execution Notes

- Follow strict order by chunk; do not start Chunk N+1 until Chunk N tests pass.
- Keep CAT-Seg code unchanged unless a compatibility shim is strictly required.
- Default seeds: `{0, 1, 2}` unless user overrides.
- For ambiguous formulas, follow the approved design-spec interpretation and log assumptions in code comments only where non-obvious.
- Preserve all per-task proxies in `work_dirs/<exp>/metrics.jsonl` from first runnable version onward.
