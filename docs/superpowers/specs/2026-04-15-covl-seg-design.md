# COVL-Seg Design Spec

**Date:** 2026-04-15
**Topic:** Implementation of "Spectral Information-Theoretic Framework for Continual Open-Vocabulary Semantic Segmentation" (COVL-Seg) per the TPAMI manuscript at `C:\Users\Mercury\Desktop\OVCL\SIGF\latex\covss-tpami.tex`.
**Status:** Draft for user + spec-review approval.

## 1. Goal and Scope

Reproduce the COVL-Seg method and experiments of the paper in a way that is faithful to Algorithm 1 and §5.1 Implementation Details, and that produces the main results table (§6.2), the ablation (§6.3), and the theory-validation analysis plots (§6.4).

**Target outcome ("Done" definition):**
- All unit tests pass (`pytest tests/`).
- End-to-end smoke run on 3090 completes without NaN.
- Full runs on a 1–2×RTX 4090 server cover: ADE20K-15, COCO-Stuff164-10, and ADE20K-100, for COVL-Seg and 5 baselines (FT, Oracle, EWC, MiB, PLOP, ZSCL), 3 seeds each.
- Main table + ablation table + theory-validation figures are generated from logged metrics.
- All empirical proxies from Table 2 of the paper are written to `metrics.jsonl` during training.
- A one-command reproduction recipe exists in the README.

**Non-goals (explicit):**
- We do not implement SSUL, WiSE-FT, or S-Prompts baselines (descoped per J1).
- We do not use any compute outside the user's 3090 (dev) and 4090 server (production).
- We do not redesign CAT-Seg internals; we treat CAT-Seg as an upstream submodule.

## 2. Relation to CAT-Seg

CAT-Seg is already cloned at `D:\COVL-Seg\CAT-Seg`. We will **not** modify CAT-Seg in place. Instead:

- A new sibling package `D:\COVL-Seg\covl_seg` contains all of our code.
- From CAT-Seg we import only:
  - CLIP wrapper from `cat_seg/modeling/transformer/` (used as the frozen `f_C^V` and the text encoder).
  - Dataset registration helpers under `cat_seg/data/datasets/` (ADE20K/COCO-Stuff/PC/VOC).
  - Class-name JSONs under `CAT-Seg/datasets/` (`ade150.json`, `coco.json`, `pc59.json`, `pc459.json`, `voc20.json`).
- CAT-Seg is kept runnable as an independent baseline for comparison and rollback.
- The whole system is built on top of Detectron2 (same as CAT-Seg), to reuse its dataset registration, trainer hooks, and evaluator plumbing.

## 3. Repository Layout

```
D:\COVL-Seg\
├── CAT-Seg\                        # upstream, unchanged
├── covl_seg\                       # our package
│   ├── __init__.py
│   ├── model\
│   │   ├── covl_seg_model.py       # top-level model: CLIP_frozen + f_S + φ + fusion
│   │   ├── continual_backbone.py   # f_S^(t): ViT-B/16 init from CLIP visual
│   │   ├── hciba_head.py           # φ = (φ_bnd, φ_sem)
│   │   ├── boundary_detect.py      # CLIP attn-gradient boundary map
│   │   └── fusion.py               # product-of-experts + α*(t) + τ_pred(t)
│   ├── losses\
│   │   ├── mine.py                 # MINE critic + Î_MINE
│   │   ├── ciba.py                 # Phase 1 boundary/semantic CIBA loss + β*
│   │   ├── ctr.py                  # CLIP text regularisation (Thm bg_entropy)
│   │   └── segmentation.py         # CE over cos(Φ, E_c) / τ
│   ├── continual\
│   │   ├── fisher.py               # implicit Fisher matvec + power iteration + k_t
│   │   ├── spectral_ogp.py         # V_τ, Ω_{τ,t}, gradient projection
│   │   └── replay_buffer.py        # SACR: priority p_i, eviction
│   ├── data\
│   │   ├── ade20k_15.py
│   │   ├── ade20k_100.py
│   │   ├── coco_stuff_164_10.py
│   │   ├── continual_loader.py     # yields (D_t, M) mixed batches
│   │   └── splits\                 # frozen task→class JSONs
│   ├── engine\
│   │   ├── trainer.py              # 4-phase loop implementing Algorithm 1
│   │   ├── evaluator.py            # mIoU_all/old/new + BG-mIoU + BWT
│   │   ├── open_vocab_eval.py      # PC-59 / PC-459 / VOC-20 zero-shot
│   │   └── hooks.py                # online proxy logging (Table 2)
│   ├── baselines\
│   │   ├── ft.py                   # COVL-Seg with everything off
│   │   ├── oracle.py               # joint training upper bound
│   │   ├── ewc.py
│   │   ├── mib.py
│   │   ├── plop.py
│   │   └── zscl.py
│   ├── configs\
│   │   ├── covl_seg_vitb_ade15.yaml
│   │   ├── covl_seg_vitb_ade100.yaml
│   │   ├── covl_seg_vitb_coco.yaml
│   │   └── covl_seg_vitl_ade15.yaml    # 4090 server only
│   ├── scripts\
│   │   ├── prepare_data.sh
│   │   ├── train_continual.py
│   │   ├── eval_continual.py
│   │   └── make_analysis_figs.py
│   └── tests\                      # pytest unit tests per module
└── docs\superpowers\specs\         # this document lives here
```

## 4. Problem Setup (recap from paper §4)

- **Frozen CLIP visual backbone** `f_C^V: R^{H×W×3} → R^{N_p × d}` (ViT-B/16, `d = 512`).
- **Continual segmentation backbone** `f_S^(t): R^{H×W×3} → R^{H×W×d_s}` (ViT-B/16 initialised from CLIP visual weights, `d_s = 512`), trained task-by-task.
- **HCIBA head** `φ = (φ_bnd, φ_sem)`: two shared linear projections `R^{d_s} → R^d`, total `2 d_s d = 2×512×512` parameters.
- **Text embeddings** `{E_c}` obtained from CLIP text encoder, precomputed and cached per dataset vocab.
- **Continual task stream**: at step `t`, data `D_t` covers new class set `C_t`; previous classes are `∪_{τ<t} C_τ`; the "background set" `B_t` is all other dataset classes (future + unlabelled existing).

At inference, pixel predictions from the CLIP stream (`p^C`) and the continual stream (`p^S`) are fused by product-of-experts with task-level scalar `α*(t)` and spatially adaptive modulation at boundary pixels, then sharpened by temperature `τ_pred(t) = c_0 (n_t Î_exc^S)^{-1/4}`.

## 5. Model Architecture

**Forward data flow (single image, task step t):**

```
x ──┬── f_C^V (frozen CLIP ViT-B/16) ──► Z_C                                (patch tokens)
    │                                 └► attention grads ──► ∂Ω_t          (boundary mask)
    │
    └── f_S^(t) (trainable ViT-B/16) ──► Z_S
                                       ├► φ_bnd(Z_S[∂])  ──► Φ_bnd
                                       └► φ_sem(Z_S)     ──► Φ_sem
                                            │
                                            └─ cos(Φ_*, E_c) ──► p^S_bnd, p^S_sem

Inference fusion per pixel (i,j):
    p(y|x)_ij ∝ softmax( (α*(t)·log p^S_ij + (1-α*(t))·log p^C_ij) / τ_pred(t) )
    with spatial modulation: boundary pixels weigh f_S more strongly via φ_bnd.
```

**Module responsibilities:**

1. **`ContinualBackbone`** — ViT-B/16 initialised from CLIP visual weights; outputs dense feature map by reshaping patch tokens to `H×W` and bilinear-upsampling to the label resolution. No CAT-Seg cost aggregation (by design, Section 2 of our brainstorm), so the model corresponds literally to `f_S` in the paper.
2. **`CLIPFrozen`** — adapter around CAT-Seg's CLIP wrapper. Exposes `encode_image`, `encode_text`, and the last-layer attention map for boundary detection.
3. **`BoundaryDetector`** — thresholds the spatial gradient of CLIP patch-level attention to obtain `∂Ω_t`; upsampled bilinearly to image resolution. Fallback (config-switchable): Sobel on RGB.
4. **`HCIBAHead`** — `φ_bnd` and `φ_sem`, each a `Linear(d_s, d)` shared across tasks.
5. **`FusionHead`** — stores `α*(t)` and `τ_pred(t)` after each task; at inference applies the product-of-experts rule with spatial modulation.

**Gradient flow control:**
- Phase 1: `f_S` frozen; only `φ` and MINE critics receive gradients.
- Phase 2: `f_S` + `φ` trained; CLIP frozen throughout.
- Enforced via `requires_grad_` toggles wrapped in a `PhaseContextManager`.

**Dimensions under ViT-L (config-parameterised):** `d_s` becomes 768, and `φ` is either `Linear(768, 768)` if text embeddings come from ViT-L, or `Linear(768, 512)` if we mix ViT-B text with ViT-L vision. Config exposes both `feat_dim_s` and `feat_dim_text`.

## 6. Algorithm 1 Realisation

### Phase 1 — HCIBA pre-alignment (`N_pre = 200` iters)

Implements Algorithm 1 lines 2–11, corresponding to Theorem 1 (CIBA) and its dense extension via Theorem 7 (granularity gap) and Corollary A.2 (optimal β*).

Per mini-batch:
```
Z_C = clip_vis(x); Z_S = f_S(x)                    # f_S frozen this phase
∂   = boundary_detector(clip_attn)

# β̂_1*, β̂_2* from Corollary A.2, with EMA(0.9) smoothing:
Δ_bnd = Î_MINE(Z_S[∂]; Y[∂]; T_S_bnd) − Î_MINE(Z_C[∂]; Y[∂]; T_C_bnd)
β̂_1* = EMA(Δ_bnd / (tr(Σ̂_S[∂])/d − 1)_+ )
# analogous for semantic stream over class-pooled features

ℓ_1 = −Î_MINE(φ_bnd(Z_S[∂]); Y[∂]) + β̂_1* · ||φ_bnd(Z_S[∂]) − Z_C[∂]||^2
ℓ_2 = −Î_MINE(φ̄_sem(Z̄_S); Y)     + β̂_2* · ||φ̄_sem(Z̄_S) − E_{c(B)}||^2
loss_phase1 = ℓ_1 + ℓ_2

update φ via AdamW; update MINE critics (T_C, T_S) via MINE loss
```

Design notes:
- MINE critics are persistent across phases and tasks (Phase 1 and Phase 3 share them).
- For critic updates, the input features are detached (critics learn MI estimation without contaminating backbone gradients). For the CIBA loss, features are not detached, so the MI term flows into `φ`.
- EMA(0.9) on both β̂_1* and β̂_2* to suppress early-iteration noise.
- `(tr(Σ̂_S)/d − 1)_+` is clipped to `+ε` to avoid division blow-up.
- β̂_2* is computed over per-segment pooled features (segment = connected component of each class in the label map).
- Per-stream estimators: Corollary A.2 in the paper gives a single `β̂*`; we split it into `β̂_1*` (boundary) and `β̂_2*` (semantic) by recomputing `Δ_S^C` and `Σ̂_S` on the boundary pixel subset and the segment-pooled subset respectively. This matches Table 2 row "β̂_1*, β̂_2* — boundary / semantic splits".

### Phase 2 — Joint backbone update (`N_main = 4000` iters)

Implements Algorithm 1 lines 12–22, corresponding to Theorems 2/8 (spectral forgetting) and 9 (background entropy catastrophe).

Per mini-batch (drawn from `D_t ∪ M` with mix ratio 3:1):
```
Z_S   = f_S(x);    Φ = φ(Z_S)
L_task = CE( cos(Φ, [E_c]_{c∈C_t}) / τ,  Y )          # MiB-style, on C_t only
# NOTE: Algorithm 1 line 14 in the paper writes `CE(Φ · [E_c], Y)` as a bare dot
# product. We interpret this as cosine-with-temperature (CLIP convention) to stay
# consistent with §5.1's inference fusion and the temperature schedule τ_pred(t).

# CTR — Theorem 9:
γ̂_clip = (2/|B_t|(|B_t|−1)) · Σ_{c≠c'∈B_t} d_TV(P_{E_c}, P_{E_{c'}})
L_CTR  = −λ_0 (1 − γ̂_clip) · Σ_{(i,j): Y_ij∈B_t} TopK-avg_{c∈B_t} cos(Φ_ij, E_c)  [K=3]

g = ∇_{f_S^attn} (L_task + L_CTR)                     # only attn-subset params
g = spectral_ogp_project(g, {V_τ}_{τ<t})              # see below
apply_grads(f_S, g); step_optimizer(φ)
```

**Spectral-OGP gradient projection (simplification, approved):**
In Phase 2 we apply the **conservative hard projection**
```
for τ in [t−1, t−2, ..., max(0, t−T_mem)]:
    g ← g − V_τ (V_τ^T g)
```
rather than the Ω-weighted variant. This is equivalent to treating `Ω_{τ,t} = Ω_max` in Algorithm 1 line 19, i.e. a strict upper bound of the paper formula. `Ω_{τ,t}` is still computed online at task boundaries for logging and for the theory-validation plots, but it does not gate Phase 2 updates.

Gradient projection is implemented via per-layer backward hooks on the attention weight subset (Q, K, V, output proj of each transformer block), using per-layer blocks of `V_τ`. We never flatten the entire `f_S` parameter vector (which would be ~350MB per vector and 18GB over 15 tasks).

### Phase 3 — Fisher subspace + fusion weight + temperature

Implements Algorithm 1 lines 23–27.

```
H_t        = implicit Fisher over params_attn using val(D_t)   # val split, not train
# NOTE: Algorithm 1 line 23 in the paper does not specify the data split used
# to build H_t. We use the held-out val split of D_t (brainstorm decision) so
# that the Fisher captures generalisation-relevant directions and is computed
# independently of the Phase 2 training batches.
k_t        = smallest k s.t. Σ_{i≤k} λ_i(H_t) ≥ (1 − ε_F) tr(H_t)    (ε_F = 0.05)
V_t        = power_iteration(H_t, k_t)           # top-k_t eigenvectors
save_fp16(V_t → work_dirs/<exp>/subspaces/V_{t:03d}.pt)

Î_exc_C, Î_exc_S = MINE_estimate(Z_C, Φ, Y; val(D_t))
α*(t)            = Î_exc_C / (Î_exc_C + Î_exc_S)
τ_pred(t)        = c_0 · (n_t · Î_exc_S)^{-1/4}        # c_0 calibrated so τ_pred(1)=0.07
```

Fisher is computed by **implicit matvec** over attention-subset parameters:
```
Hv = (1/n) Σ_i ⟨g_i, v⟩ · g_i
```
where `g_i = ∇_{f_S^attn} log p(y_i|x_i)`. Top-50 eigenvectors are extracted via power iteration with deflation, `k_t` is chosen by the 95%-energy rule, then truncated. Everything in FP32 for numerical stability even under AMP training.

Storage: `V_t ∈ R^{|attn_params| × k_t}`, FP16, ~18GB total over 15 ADE20K-15 tasks.

### Phase 4 — SACR replay buffer update

Implements Algorithm 1 lines 28–31, corresponding to Theorem 10 (compositional interference).

```
IoU_c^cur for c ∈ C_{t−1}         # eval current f_S on M per-class
Ē_t = mean_{c ∈ C_t^new} E_c
p_i = σ( cos(E_{c_i}, Ē_t) · Π̂_{c_i,t} ) · (1 − IoU_{c_i}^cur)
add top-p_i samples from D_t to M
evict lowest-priority if |M| > M_max (200 per class)
```

- `Π̂_{c_i,t}` is precomputed dataset co-occurrence frequency (one scan of the training set per dataset, cached JSON).
- Buffer stores image paths + label paths only (not raw pixels), loaded lazily by the `ContinualDataLoader`.

## 7. Data and Continual Splits

| Dataset | Classes | Steps | Classes/step | Use |
|---|---|---|---|---|
| ADE20K-15 | 150 | 15 | 10 | Primary result |
| ADE20K-100 | 150 | 100 | 5 | Long-horizon forgetting |
| COCO-Stuff164-10 | 164 | 10 | ~16 | Denser co-occurrence regime |

Protocol: **MiB disjoint** — supervision on `C_t` only; old-class pixels appear unlabelled; future-class pixels appear as background. Class orderings follow the MiB/PLOP standard (alphabetical for ADE20K, id-order for COCO-Stuff), frozen into `covl_seg/data/splits/*.json` for reproducibility.

Data preparation scripts reuse CAT-Seg's `prepare_ade20k_150.py`, `prepare_ade20k_full.py`, `prepare_coco_stuff.py`, and the existing `ade150.json` / `coco.json` class lists — no rewrites.

`ContinualDataLoader` yields mixed mini-batches of `D_t` and `M` at a 3:1 ratio (config-exposed).

## 8. Training Engine

Extends Detectron2's `DefaultTrainer` with an outer task loop and a phase controller:

```
for t in range(T):
    build_task_loader(t); freeze(f_C, text_enc)

    # Phase 1
    freeze(f_S); unfreeze(φ, MINE)
    for step in range(N_pre):  run_phase1(batch)

    # Phase 2
    unfreeze(f_S); unfreeze(φ)
    for step in range(N_main): run_phase2(batch)    # with Spectral-OGP hooks

    # Phase 3
    H_t, k_t, V_t = build_fisher_subspace(f_S, val(D_t))
    save V_t (fp16)
    Î_exc_C, Î_exc_S = mine_estimate(val(D_t))
    α_star[t], τ_pred[t] = compute_fusion_params(...)

    # Phase 4
    update_replay_buffer(D_t, M, ...)

    # Eval
    metrics_t = evaluator.run(model, seen_classes_t)
    open_vocab_metrics_t = open_vocab_evaluator.run(model, PC59 / PC459 / VOC20)
    log_all(metrics_t, open_vocab_metrics_t, proxies_t)
    save_checkpoint(t)
```

- Phase switching and proxy logging are implemented as Detectron2 `HookBase` subclasses.
- AMP enabled for Phases 1–2; disabled (FP32) inside the Fisher/power-iteration code path for numerical stability.
- Checkpointing at the end of each task: `f_S`, `φ`, `V_t`, `M` (indices only), `α*(t)`, `τ_pred(t)`, `metrics_t.json`. Resume-from-task-`t` supported.

## 9. Evaluation

At the end of each task `t`:
- **Closed-set continual metrics** on the held-out val split, over seen classes `∪_{τ≤t} C_τ`:
  - `mIoU^all_t`, `mIoU^old_t`, `mIoU^new_t`, `BG-mIoU_t`.
- **BWT** after the final task: `(1/(T−1)) Σ_{τ=1}^{T−1} (IoU_τ^(τ) − IoU_τ^(T))`, using the per-class IoU matrix built incrementally across tasks.
- **Open-vocabulary zero-shot evaluation** (J2): run the current model (using `φ` + fusion) on PC-59, PC-459, and VOC-20 by encoding their class names with CLIP text encoder; no fine-tuning. Logged per task step as `ov_mIoU_{dataset}_t`.

All metrics and proxies are written to `work_dirs/<exp>/metrics.jsonl` (one JSON object per task-step record), plus a final CSV/markdown summary table.

## 10. Baselines (J1 = option c)

| Baseline | Implementation strategy |
|---|---|
| **FT** | COVL-Seg with HCIBA off, Spectral-OGP off, CTR off, replay off. |
| **Oracle** | COVL-Seg trained jointly on `∪_t D_t` for one long run, upper bound. |
| **EWC** | FT + Fisher regulariser, reusing `continual/fisher.py`. ~100 LOC wrapper. |
| **MiB** | Our trainer + MiB unbiased CE loss; reference implementation from MiB's public repo (loss only). |
| **PLOP** | Our trainer + PoD loss + entropy pseudo-label; ported from PLOP's public repo (loss module only). |
| **ZSCL** | Our trainer + CLIP feature distillation loss; ported from ZSCL's public repo. |

Fallback (documented in README if public code is unusable): re-implement each method's core loss from the paper (budget 1–2 days per baseline), or further descope to `{FT, Oracle, EWC, MiB, PLOP}` if ZSCL port blocks progress.

## 11. Empirical Proxies (paper Table 2)

Written to `metrics.jsonl` by `engine/hooks.py`:

| Quantity | Update frequency | Source |
|---|---|---|
| `Î_exc^C`, `Î_exc^S` | per task | MINE critics on `val(D_t)` |
| `β̂_1*`, `β̂_2*` | per task (+ EMA during Phase 1) | Corollary A.2 estimator |
| `Ω_{τ,t}` | per task pair | `||V_τ^T V_t||_F^2 / sqrt(k_τ k_t)` |
| `γ̂_clip(t)` | per task | pairwise TV of CLIP text softmax over `B_t` |
| `δ_α(t)` | per task | `G_t^boundary / (|Ω|(Î_exc^C + Î_exc^S))` |
| `T̂_{τ,t}` | per task | `σ(cos(E_τ, Ē_t) · Π̂_{τt})` |
| `Π̂_{τ,t}` | precomputed | dataset co-occurrence scan |
| `τ_pred(t)` | per task | `c_0 (n_t Î_exc^S)^{-1/4}` |

These drive the theory-validation plots in the final figure script `scripts/make_analysis_figs.py`.

## 12. Hyperparameters (fixed from paper §5.1)

- Optimizer: AdamW, lr `3e-4`, weight decay `1e-2`, cosine schedule.
- `N_pre = 200`, `N_main = 4000` per task.
- `λ_0 = 0.1`, `K = 3` (CTR TopK), `ε_F = 0.05`, `T_mem = all tasks`, `M_max = 200` per class.
- `c_0` calibrated so `τ_pred(1) = 0.07`.
- CLIP: ViT-B/16; `d = d_s = 512`.
- MINE critics `T_C`, `T_S`: two-layer MLPs `512 → 256 → 1` with GELU activations (paper §5.1).
- Effective batch size 32 (gradient accumulation on 1–2 4090s).

All hyperparameters fixed across datasets per the paper (no task-specific tuning).

## 13. Testing Strategy

- `pytest` unit tests per module, covering:
  - **Numerical correctness**: Fisher power iteration vs `torch.linalg.eigh` on small (100×100) synthetic matrices; MINE MI estimate on Gaussian pairs with known analytical MI; β* closed-form on hand-built toy input.
  - **Invariants**: Spectral-OGP post-projection `||V_τ^T g||₂ ≈ 0`; replay buffer priority ordering; continual split coverage.
  - **Shape/gradient sanity**: every `nn.Module` forward on dummy `(1,3,384,384)` input and backward without NaNs.
  - **Dataset integrity**: per task the label map contains only `C_t` (old/future suppressed correctly); union of all `C_t` equals full class set.
  - **End-to-end smoke**: 1 task × 5 iters × CPU (regression guard against NaN/crash).
- Unit tests run on the 3090 in seconds; smoke test ≤5 minutes.

## 14. Implementation Milestones

```
M0  repo skeleton + CAT-Seg import + detectron2 env sanity
M1  model layer (continual backbone, boundary detect, HCIBA head, fusion, main model)
M2  loss layer (MINE, CIBA, CTR, segmentation CE)
M3  continual mechanisms (Fisher, Spectral-OGP, replay buffer)
M4  data layer (3 continual splits + loader + splits JSON + co-occurrence precompute)
M5  engine (trainer 4-phase loop, evaluator, hooks)
M6  3090 local smoke (ADE20K-15 first 3 tasks × 400 iters, ViT-B)
M7  baselines (FT, Oracle, EWC, MiB, PLOP, ZSCL)
M8  open-vocab zero-shot evaluator (PC-59/PC-459/VOC-20)
M9  4090 server production runs (3 datasets × 6 methods × 3 seeds)
M10 theory-validation figures + main/ablation tables + README reproduction recipe
```

3090 runs M0–M8 up to smoke; M9 is 4090-only.

## 15. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CLIP attention gradient boundary detection too noisy | HCIBA boundary stream ineffective | Config-switchable Sobel-on-RGB fallback. |
| MINE training unstable | β* oscillates, Phase 1 diverges | EMA(0.9) smoothing; double `N_pre` if needed; MINE loss clipping. |
| Fisher power iteration slow on ViT | Phase 3 bottleneck | Attn-only subspace (already chosen); top-50 eigenvectors + deflation; GPU kernel. |
| Spectral-OGP disk footprint | 15–100 task subspace store | FP16 storage; attn-only (already chosen); optional checkpoint compression. |
| Baseline code unavailable | M7 budget blow-up | Descope to {FT, Oracle, EWC, MiB, PLOP}; document in README. |
| ADE20K-100 full budget too long | M9 delay | Drop to `N_main=2000` as "compute-constrained" variant; keep ADE20K-15 as headline. |
| AMP + gradient projection causes NaN | Phase 2 crash | Force FP32 inside Fisher/power-iter; `GradScaler` unscale before OGP projection. |
| Formula ambiguity at implementation time | Blocks coding | Follow paper §5 prose as tiebreaker; annotate ambiguity in code comments; ask user if still unclear. |

## 16. Decisions Ledger (from brainstorming)

| ID | Decision | Rationale |
|---|---|---|
| D1 | Range = full reproduction (Option A) | User confirmed. |
| D2 | Segmentation framework = CAT-Seg as submodule, not in-place edit | Clean baseline; upstream isolation. |
| D3 | Trainer stack = Detectron2 | CAT-Seg is D2, reuse infra. |
| D4 | `f_S` = pure ViT-B/16 with token upsample, no CAT-Seg cost aggregation | Literal match to paper formulas. |
| D5 | Phase 2 Spectral-OGP = hard projection (`Ω` logged only) | Conservative upper bound of paper, avoids Phase-3 dependency during Phase 2. |
| D6 | Fisher subspace = attention-subset parameters only | EWC/OGP precedent; disk and compute friendly. |
| D7 | β* smoothing = EMA(0.9) | Suppress early MINE noise. |
| D8 | `V_τ` stored in FP16 on disk | Halve footprint; numerics still sufficient. |
| D9 | Baselines = {FT, Oracle, EWC, MiB, PLOP, ZSCL} (J1 = c) | Cost/coverage tradeoff. |
| D10 | Add open-vocab zero-shot eval (J2 = b) | Supports "open-vocabulary" claim in title. |
| D11 | Full budget on ADE20K-100 (J3 = full) | Paper-faithful long-horizon result. |

## 17. Open Questions for User Review

1. Is the ViT-L config (`covl_seg_vitl_ade15.yaml`) required for the initial milestones, or can it be added later only if 4090 compute allows it?
2. For ADE20K-100, is it acceptable to launch the full-budget run only after ADE20K-15 and COCO-Stuff164-10 are fully validated (to minimise blast radius of a potential bug surfacing deep into training)?
3. Do you have a preference on the random seeds for the 3-seed protocol (e.g. `{0,1,2}` vs the MiB-convention `{1234,5678,9999}`)?

These are not blockers; I will adopt defaults (add ViT-L config later; run ADE20K-100 last; seeds = `{0,1,2}`) unless you say otherwise.
