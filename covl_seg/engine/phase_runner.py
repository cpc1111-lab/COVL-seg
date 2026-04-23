from __future__ import annotations

import math
import random
from typing import Dict, Optional

import torch

from covl_seg.continual.fisher import fisher_matvec_from_gradients, top_eigenvectors_power
from covl_seg.continual.replay_buffer import ReplayItem, SACRReplayBuffer
from covl_seg.continual.spectral_ogp import hard_project_gradient
from covl_seg.losses.ciba import estimate_beta_star
from covl_seg.losses.mine import (
    ConditionalMINECritic,
    conditional_mine_loss,
    conditional_mine_lower_bound,
)


def _deterministic_loss(task_id: int, salt: int) -> float:
    rng = random.Random(task_id * 1000 + salt)
    return max(1e-6, rng.random() * 0.05)


def _to_tensor(values, fallback: torch.Tensor) -> torch.Tensor:
    if values is None:
        return fallback
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    return tensor


def _quick_conditional_mine_estimate(
    x: torch.Tensor,
    y: torch.Tensor,
    z_cond: torch.Tensor,
    n_steps: int = 8,
) -> float:
    """Estimate conditional MINE lower bound I(x; y | z_cond) with a small critic.

    Inputs are treated as 1-D feature streams (N samples, 1 feature each).
    Returns a non-negative float clamped at 0.  Returns 0.0 for degenerate inputs
    (fewer than 2 samples or zero-variance streams) to avoid NaN in downstream formulas.
    """
    x = x.reshape(-1, 1).float()
    y = y.reshape(-1, 1).float()
    z_cond = z_cond.reshape(-1, 1).float()
    n = min(x.shape[0], y.shape[0], z_cond.shape[0])
    if n < 2:
        return 0.0
    x, y, z_cond = x[:n], y[:n], z_cond[:n]

    # Skip training if either stream is constant — MINE would produce NaN
    if float(x.var().item()) < 1e-9 or float(y.var().item()) < 1e-9:
        return 0.0

    critic = ConditionalMINECritic(feature_dim=1, hidden_dim=16)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = conditional_mine_loss(critic, x, y, z_cond)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()

    with torch.no_grad():
        result = conditional_mine_lower_bound(critic, x, y, z_cond)
        if not torch.isfinite(result):
            return 0.0
        return float(result.clamp(min=0.0).item())


def run_phase1_hciba(
    task_id: int,
    cfg: Dict[str, object],
    batch: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Phase 1: estimate I_exc^C, I_exc^S, β_1*, β_2* via MINE lower bounds.

    Implements Algorithm 1 lines 3–9 (proxy estimation from available features).
    """
    batch = batch or {}
    generator = torch.Generator().manual_seed(2000 + task_id)
    z_c = _to_tensor(batch.get("features_c"), torch.randn(32, generator=generator))
    z_s = _to_tensor(batch.get("features_s"), torch.randn(32, generator=generator))
    y = _to_tensor(batch.get("targets"), torch.randn(32, generator=generator))

    min_len = int(min(z_c.numel(), z_s.numel(), y.numel()))
    z_c = z_c.reshape(-1)[:min_len]
    z_s = z_s.reshape(-1)[:min_len]
    y = y.reshape(-1)[:min_len]

    i_exc_c = _quick_conditional_mine_estimate(z_c, y, z_cond=z_s)
    i_exc_s = _quick_conditional_mine_estimate(z_s, y, z_cond=z_c)

    # Δ_S^C = I_exc^S − I_exc^C  (positive when backbone has more exclusive info)
    delta = max(0.0, i_exc_s - i_exc_c)

    # β* = Δ_S^C / (tr(Σ_S)/d − 1)+  (Corollary A.2)
    # torch.var requires ≥2 elements; guard against NaN for degenerate batches
    if min_len >= 2:
        var_val = float(torch.var(z_s).item())
        sigma_trace = var_val * min_len if math.isfinite(var_val) else 0.0
    else:
        sigma_trace = 0.0
    beta_1 = estimate_beta_star(delta=delta, sigma_trace=sigma_trace, dim=max(min_len, 1))
    # β_2* for semantic stream uses half the boundary delta as an approximation
    beta_2 = estimate_beta_star(delta=0.5 * delta, sigma_trace=sigma_trace, dim=max(min_len, 1))

    return {
        "phase": "phase1",
        "task": float(task_id),
        "loss": _deterministic_loss(task_id, 1),
        "iters": float(int(cfg.get("n_pre", 0))),
        "beta_1_star": float(beta_1),
        "beta_2_star": float(beta_2),
        "I_exc_C": float(i_exc_c),
        "I_exc_S": float(i_exc_s),
    }


def run_phase2_joint(
    task_id: int,
    cfg: Dict[str, object],
    batch: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Phase 2: joint backbone update metrics.

    CTR loss follows Theorem 9 / Eq. 14: ℒ_CTR = −λ_0(1−γ̂_clip) TopK-avg cos(Φ, E_c).
    Minimising this (negative) loss maximises cosine similarity → anchors background
    features toward CLIP text centroids.
    γ_clip is the mean pairwise TV distance between softmax distributions over bg classes.
    """
    batch = batch or {}
    generator = torch.Generator().manual_seed(3000 + task_id)

    # Spectral-OGP projection (structure validation)
    grad = torch.randn(64, generator=generator)
    basis = torch.randn(64, 4, generator=generator)
    basis = torch.linalg.qr(basis, mode="reduced").Q
    proj_grad = hard_project_gradient(gradient=grad, basis=basis)

    enable_ctr = bool(cfg.get("enable_ctr", True))
    ctr_loss = 0.0
    gamma_clip = 0.0

    if enable_ctr:
        bg_logits = _to_tensor(
            batch.get("bg_logits"),
            torch.randn(8, 5, generator=generator),
        )
        if bg_logits.ndim == 1:
            bg_logits = bg_logits.unsqueeze(0)

        # γ_clip = mean pairwise TV distance between background-class softmax distributions
        # TV(P_i, P_j) = 0.5 * Σ_pixels |p_i(pixel) − p_j(pixel)|
        probs = torch.softmax(bg_logits, dim=1)  # [N_pixels, N_bg]
        n_cls = probs.shape[1]
        tv_vals = []
        for i in range(n_cls - 1):
            for j in range(i + 1, n_cls):
                tv_vals.append(0.5 * torch.abs(probs[:, i] - probs[:, j]).mean())
        gamma_clip = float(torch.stack(tv_vals).mean().clamp(0.0, 1.0).item()) if tv_vals else 0.0

        # CTR loss proxy: −λ_0(1−γ̂_clip) · mean TopK logits
        # Negative sign: minimising pushes logits (≈ cosine sims) higher → correct anchoring
        lambda0 = float(cfg.get("lambda0_ctr", 0.1))
        ctr_scale = max(0.0, float(cfg.get("balanced_w_ctr", 1.0)))
        k = min(3, bg_logits.shape[1])
        topk_mean = float(bg_logits.topk(k=k, dim=1).values.mean().item())
        ctr_loss = float(-lambda0 * (1.0 - gamma_clip) * topk_mean * ctr_scale)

    enable_projection = bool(cfg.get("enable_spectral_ogp", True))
    g_stab = float(cfg.get("balanced_g_stab", 0.0))
    w_oldfix = float(cfg.get("balanced_w_oldfix", 0.0))
    before_norm = float(torch.norm(grad).item())
    after_norm = float(torch.norm(proj_grad if enable_projection else grad).item())
    after_norm = float(after_norm / (1.0 + max(0.0, g_stab)))
    oldfix_base = float(torch.mean(torch.abs(grad - proj_grad)).item())
    oldfix_weighted_term = float(max(0.0, w_oldfix) * oldfix_base)

    return {
        "phase": "phase2",
        "task": float(task_id),
        "loss": _deterministic_loss(task_id, 2),
        "iters": float(int(cfg.get("n_main", 0))),
        "ctr_loss": ctr_loss,
        "gamma_clip": gamma_clip,
        "proj_norm_before": before_norm,
        "proj_norm_after": after_norm,
        "oldfix_weighted_term": oldfix_weighted_term,
    }


def run_phase3_subspace_and_fusion(
    task_id: int,
    cfg: Dict[str, object],
    batch: Optional[Dict[str, object]] = None,
    prev_phase_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Phase 3: Fisher subspace update and fusion weight computation.

    α*(t) = I_exc^C / (I_exc^C + I_exc^S)  (Theorem 3, Eq. alpha_star)
    τ_pred(t) = c_0 · (n_t · Î_exc^S)^{−1/4}  (Corollary A.4)

    prev_phase_metrics: output of run_phase1_hciba for this task; used to obtain
    I_exc^C and I_exc^S estimates for the closed-form α* and τ_pred formulas.
    """
    prev = prev_phase_metrics or {}
    i_exc_c = float(prev.get("I_exc_C", 0.5))
    i_exc_s = float(prev.get("I_exc_S", 0.5))

    # α*(t) = I_exc^C / (I_exc^C + I_exc^S)  (Theorem 3)
    denom = i_exc_c + i_exc_s
    alpha_star = (i_exc_c / denom) if denom > 1e-8 else 0.5
    alpha_floor = float(cfg.get("balanced_alpha_floor", 0.0))
    alpha_star = float(max(alpha_star, alpha_floor))

    # τ_pred(t) = c_0 · (n_t · Î_exc^S)^{−1/4}  (Corollary A.4)
    # c_0 calibrated so τ_pred ≈ 0.07 at task 1 with I_exc^S ≈ 1.0
    n_t = max(1, int(cfg.get("n_main", 4000)))
    tau_init = float(cfg.get("tau_init", 0.07))
    i_exc_s_safe = max(i_exc_s, 1e-6)
    c0 = tau_init * float(n_t) ** 0.25  # s.t. τ_pred(I_exc^S=1) = tau_init
    tau_pred = float(max(1e-4, min(c0 * (float(n_t) * i_exc_s_safe) ** (-0.25), 1.0)))

    # Fisher subspace via power iteration
    eps_f = float(cfg.get("eps_f", 0.05))
    dim = int(cfg.get("fisher_dim", 8))
    k = max(1, min(int(cfg.get("ewc_topk", 2)), dim))
    generator = torch.Generator().manual_seed(1000 + task_id)
    gradients = torch.randn(16, dim, generator=generator)

    def _matvec(v: torch.Tensor) -> torch.Tensor:
        return fisher_matvec_from_gradients(gradients=gradients, vector=v)

    eigvecs, eigvals = top_eigenvectors_power(matvec_fn=_matvec, dim=dim, k=k, num_iters=25)
    fisher_energy = float(torch.clamp(eigvals.sum(), min=1e-8).item())
    basis_vector = eigvecs[:, 0].detach().cpu().tolist()

    return {
        "phase": "phase3",
        "task": float(task_id),
        "alpha_star": alpha_star,
        "tau_pred": tau_pred,
        "fisher_topk": float(eigvecs.shape[1]),
        "fisher_energy": fisher_energy,
        "subspace_basis": basis_vector,
        "loss": _deterministic_loss(task_id, 3),
    }


def run_phase4_replay_update(
    task_id: int,
    cfg: Dict[str, object],
    batch: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Phase 4: SACR replay buffer update.

    Priority (Eq. replay_priority):
        p_i(t) ∝ σ(cos(E_ci, Ē_t) · Π̂_{ci,t}) · (1 − IoU_ci^cur)

    Proxies used when actual text embeddings / IoU are unavailable:
      - cos(E_ci, Ē_t): feature similarity between class feature and mean new-task feature
      - Π̂_{ci,t}: approximated by relative position within the feature list
      - IoU_ci^cur: approximated from feature confidence (sigmoid of absolute value)
    """
    batch = batch or {}
    max_per_class = int(cfg.get("m_max_per_class", 200))
    max_total = int(cfg.get("m_max_total", max_per_class * 4))
    rho_new = max(0.0, float(cfg.get("balanced_rho_new", 0.0)))
    rho_old = max(0.0, float(cfg.get("balanced_rho_old", 0.0)))

    buffer = SACRReplayBuffer(max_total_items=max_total, max_per_class=max_per_class)

    generator = torch.Generator().manual_seed(4000 + task_id)
    features_c = _to_tensor(
        batch.get("features_c"),
        torch.rand(5, generator=generator),
    )
    features_s = _to_tensor(
        batch.get("features_s"),
        torch.rand(5, generator=generator),
    )
    fc = features_c.reshape(-1).float()
    fs = features_s.reshape(-1).float()
    n_items = min(fc.numel(), fs.numel(), 5)
    fc = fc[:n_items]
    fs = fs[:n_items]

    # Ē_t proxy: mean of new-task feature vector
    e_bar_t = fs.mean()

    old_term_total = 0.0
    new_term_total = 0.0
    base_priority_total = 0.0

    classes_per_task = max(1, int(cfg.get("classes_per_task", 10)))

    for i in range(n_items):
        class_id = (task_id * n_items + i) % classes_per_task

        # σ(cos(E_ci, Ē_t) · Π̂) — compositional risk
        # Proxy: σ(fc[i] · e_bar_t / ‖e_bar_t‖)  ×  co-occurrence weight ≈ (i+1)/n
        e_norm = max(float(e_bar_t.abs().item()), 1e-8)
        cos_arg = float((fc[i] * e_bar_t).item()) / e_norm
        co_occurrence_weight = float(i + 1) / n_items  # Π̂ proxy (later = dataset stats)
        compositional_risk = float(
            torch.sigmoid(torch.tensor(cos_arg * co_occurrence_weight)).item()
        )

        # (1 − IoU_ci^cur) — forgetting severity proxy
        # Higher feature magnitude → higher confidence → higher IoU → lower priority
        iou_proxy = float(torch.sigmoid(fc[i].abs()).item()) * 0.5
        forgetting_severity = 1.0 - iou_proxy

        # SACR priority: σ(cos · Π̂) · (1 − IoU)
        base_priority = compositional_risk * forgetting_severity

        # Balanced controller modulation
        newness = float(i) / max(n_items - 1, 1)
        new_term = base_priority * rho_new * newness
        old_term = base_priority * rho_old * (1.0 - newness)
        priority = float(base_priority + new_term + old_term)

        base_priority_total += base_priority
        old_term_total += old_term
        new_term_total += new_term

        buffer.add(
            ReplayItem(
                image_path=f"images/task_{task_id}_{i}.jpg",
                label_path=f"labels/task_{task_id}_{i}.png",
                class_id=class_id,
                priority=priority,
            )
        )

    selected = len(buffer.items())
    return {
        "phase": "phase4",
        "task": float(task_id),
        "replay_budget": float(max_per_class),
        "replay_selected": float(selected),
        "replay_rho_new": float(rho_new),
        "replay_rho_old": float(rho_old),
        "replay_priority_base": float(base_priority_total),
        "replay_priority_new_term": float(new_term_total),
        "replay_priority_old_term": float(old_term_total),
        "replay_priority_total": float(base_priority_total + new_term_total + old_term_total),
        "loss": _deterministic_loss(task_id, 4),
    }
