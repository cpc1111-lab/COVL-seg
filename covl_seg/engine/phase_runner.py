from __future__ import annotations

import math
import random
from typing import Dict, Optional

import torch

from covl_seg.continual.fisher import fisher_matvec_from_gradients, top_eigenvectors_power
from covl_seg.continual.replay_buffer import ReplayItem, SACRReplayBuffer
from covl_seg.continual.spectral_ogp import hard_project_gradient


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


def run_phase1_hciba(task_id: int, cfg: Dict[str, object], batch: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    generator = torch.Generator().manual_seed(2000 + task_id)
    z_c = _to_tensor((batch or {}).get("features_c"), torch.randn(32, generator=generator))
    z_s = _to_tensor((batch or {}).get("features_s"), torch.randn(32, generator=generator))
    y = _to_tensor((batch or {}).get("targets"), torch.randn(32, generator=generator))

    min_len = int(min(z_c.numel(), z_s.numel(), y.numel()))
    z_c = z_c.reshape(-1)[:min_len]
    z_s = z_s.reshape(-1)[:min_len]
    y = y.reshape(-1)[:min_len]

    i_exc_c = float(torch.nn.functional.cosine_similarity(z_c, y, dim=0).abs().item())
    i_exc_s = float(torch.nn.functional.cosine_similarity(z_s, y, dim=0).abs().item())
    delta = max(0.0, i_exc_s - i_exc_c)
    variance_proxy = max(1e-6, float(torch.var(z_s).item()))
    beta_1 = delta / variance_proxy
    beta_2 = 0.5 * beta_1
    return {
        "phase": "phase1",
        "task": float(task_id),
        "loss": _deterministic_loss(task_id, 1),
        "iters": float(int(cfg.get("n_pre", 0))),
        "beta_1_star": float(beta_1),
        "beta_2_star": float(beta_2),
        "I_exc_C": i_exc_c,
        "I_exc_S": i_exc_s,
    }


def run_phase2_joint(task_id: int, cfg: Dict[str, object], batch: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    generator = torch.Generator().manual_seed(3000 + task_id)
    grad = torch.randn(64, generator=generator)
    basis = torch.randn(64, 4, generator=generator)
    basis = torch.linalg.qr(basis, mode="reduced").Q
    proj_grad = hard_project_gradient(gradient=grad, basis=basis)

    enable_ctr = bool(cfg.get("enable_ctr", True))
    ctr_loss = 0.0
    gamma_clip = 0.0
    if enable_ctr:
        bg_logits = _to_tensor((batch or {}).get("bg_logits"), torch.randn(16, 5, generator=generator))
        if bg_logits.ndim == 1:
            bg_logits = bg_logits.unsqueeze(0)
        ctr_loss = float(torch.topk(bg_logits, k=3, dim=1).values.mean().abs().item())
        probs = torch.softmax(bg_logits, dim=1)
        tv_vals = []
        for i in range(probs.shape[0] - 1):
            tv_vals.append(0.5 * torch.abs(probs[i] - probs[i + 1]).sum())
        if tv_vals:
            gamma_clip = float(torch.stack(tv_vals).mean().clamp(0.0, 1.0).item())
        else:
            gamma_clip = 0.0

    enable_projection = bool(cfg.get("enable_spectral_ogp", True))
    before_norm = float(torch.norm(grad).item())
    after_norm = float(torch.norm(proj_grad if enable_projection else grad).item())

    return {
        "phase": "phase2",
        "task": float(task_id),
        "loss": _deterministic_loss(task_id, 2),
        "iters": float(int(cfg.get("n_main", 0))),
        "ctr_loss": ctr_loss,
        "gamma_clip": gamma_clip,
        "proj_norm_before": before_norm,
        "proj_norm_after": after_norm,
    }


def run_phase3_subspace_and_fusion(task_id: int, cfg: Dict[str, object], batch: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    eps_f = float(cfg.get("eps_f", 0.05))
    alpha_star = 1.0 / (1.0 + math.exp(-0.2 * task_id))
    tau_pred = 0.07 * (1.0 + eps_f) / (task_id ** 0.25)

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
        "alpha_star": float(alpha_star),
        "tau_pred": float(tau_pred),
        "fisher_topk": float(eigvecs.shape[1]),
        "fisher_energy": fisher_energy,
        "subspace_basis": basis_vector,
        "loss": _deterministic_loss(task_id, 3),
    }


def run_phase4_replay_update(task_id: int, cfg: Dict[str, object], batch: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    max_per_class = int(cfg.get("m_max_per_class", 200))
    max_total = int(cfg.get("m_max_total", max_per_class * 4))
    buffer = SACRReplayBuffer(max_total_items=max_total, max_per_class=max_per_class)
    for i in range(1, 6):
        class_id = (task_id + i) % 4
        priority = float(1.0 / i)
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
        "loss": _deterministic_loss(task_id, 4),
    }
