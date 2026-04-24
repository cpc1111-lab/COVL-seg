from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from covl_seg.continual.task_partition import TaskDef
from covl_seg.continual.spectral_ogp import hard_project_gradient
from covl_seg.engine.phase_runner import run_phase4_replay_update
from covl_seg.losses.ciba import ciba_alignment_loss, estimate_beta_star
from covl_seg.losses.mine import ConditionalMINECritic, conditional_mine_loss, conditional_mine_lower_bound


def _make_synthetic_batch(
    task: TaskDef,
    num_classes: int,
    *,
    batch_size: int,
    image_size: int,
    generator: torch.Generator,
    use_seen_classes: bool,
) -> Dict[str, torch.Tensor]:
    class_ids = task.seen_classes if use_seen_classes else task.new_classes
    if not class_ids:
        class_ids = task.seen_classes or list(range(num_classes))
    if not class_ids:
        class_ids = [0]

    class_pool = torch.tensor(class_ids, dtype=torch.long)
    sampled = torch.randint(0, class_pool.shape[0], (batch_size, image_size, image_size), generator=generator)
    targets = class_pool[sampled]

    idx = torch.arange(num_classes, dtype=torch.float32)
    palette = torch.stack([torch.sin(idx), torch.cos(idx), torch.tanh(idx / 10.0)], dim=1)
    images = palette[targets] + 0.05 * torch.randn(batch_size, image_size, image_size, 3, generator=generator)
    images = images.permute(0, 3, 1, 2).contiguous().float()
    return {"images": images, "targets": targets.long()}


def _compute_gamma_clip(bg_logits: torch.Tensor) -> float:
    if bg_logits.numel() == 0:
        return 0.0
    if bg_logits.ndim == 1:
        bg_logits = bg_logits.unsqueeze(0)
    probs = torch.softmax(bg_logits, dim=1)
    n_cols = probs.shape[1]
    if n_cols <= 1:
        return 0.0
    tv_vals: List[torch.Tensor] = []
    for i in range(n_cols - 1):
        for j in range(i + 1, n_cols):
            tv_vals.append(0.5 * torch.abs(probs[:, i] - probs[:, j]).mean())
    if not tv_vals:
        return 0.0
    return float(torch.stack(tv_vals).mean().clamp(0.0, 1.0).item())


def _compute_ogp_basis(model: nn.Module, task: TaskDef, cfg: Dict[str, object], seed: int) -> torch.Tensor:
    del task
    param_dim = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    history = cfg.get("_basis_history", [])
    vectors: List[torch.Tensor] = []
    if isinstance(history, list):
        for raw in history:
            try:
                candidate = torch.tensor(raw, dtype=torch.float32).reshape(-1)
            except Exception:
                continue
            if candidate.numel() == param_dim:
                vectors.append(candidate)

    if not vectors:
        g = torch.Generator().manual_seed(seed + 9000)
        fallback = torch.randn(param_dim, generator=g)
        vectors.append(fallback)

    k = max(1, min(int(cfg.get("ewc_topk", 1)), len(vectors)))
    basis = torch.stack(vectors[:k], dim=1)
    basis = torch.linalg.qr(basis, mode="reduced").Q
    return basis


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
        else:
            chunks.append(p.grad.reshape(-1))
    if not chunks:
        return torch.zeros(1)
    return torch.cat(chunks)


def _assign_flat_grads(model: nn.Module, flat_grad: torch.Tensor) -> None:
    cursor = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        width = p.numel()
        block = flat_grad[cursor : cursor + width].view_as(p)
        if p.grad is None:
            p.grad = block.clone()
        else:
            p.grad.copy_(block)
        cursor += width


def _quick_conditional_mi(x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor, steps: int = 5) -> float:
    x = x.detach().reshape(-1, 1).float()
    y = y.detach().reshape(-1, 1).float()
    cond = cond.detach().reshape(-1, 1).float()
    n = min(x.shape[0], y.shape[0], cond.shape[0])
    if n < 2:
        return 0.0
    x, y, cond = x[:n], y[:n], cond[:n]
    if float(x.var().item()) < 1e-9 or float(y.var().item()) < 1e-9:
        return 0.0

    critic = ConditionalMINECritic(feature_dim=1, hidden_dim=16).to(device=x.device, dtype=x.dtype)
    opt = torch.optim.Adam(critic.parameters(), lr=5e-3)
    for _ in range(steps):
        opt.zero_grad()
        mine_obj = conditional_mine_loss(critic, x, y, cond)
        if not torch.isfinite(mine_obj):
            break
        mine_obj.backward()
        opt.step()

    with torch.no_grad():
        mi = conditional_mine_lower_bound(critic, x, y, cond)
        if not torch.isfinite(mi):
            return 0.0
        return float(mi.clamp(min=0.0).item())


def _build_replay_batch(outputs: Dict[str, torch.Tensor], targets: torch.Tensor, bg_logits: torch.Tensor) -> Dict[str, object]:
    projected = outputs["projected"]
    boundary = outputs["boundary_map"]
    features_c = projected.mean(dim=(2, 3)).reshape(-1)
    features_s = boundary.mean(dim=(2, 3)).reshape(-1)
    targets_1d = targets.float().reshape(-1)
    return {
        "features_c": [float(x) for x in features_c.detach().cpu().tolist()],
        "features_s": [float(x) for x in features_s.detach().cpu().tolist()],
        "targets": [float(x) for x in targets_1d.detach().cpu().tolist()[: features_c.numel()]],
        "bg_logits": bg_logits.detach().cpu().tolist(),
    }


def _emit_progress(
    progress_callback: Optional[Callable[[Dict[str, object]], None]],
    *,
    phase: str,
    current: int,
    total: int,
    message: Optional[str] = None,
) -> None:
    if progress_callback is None:
        return
    payload: Dict[str, object] = {
        "phase": phase,
        "current": int(current),
        "total": int(max(total, 1)),
    }
    if message:
        payload["message"] = message
    progress_callback(payload)


def run_mock_task_training(
    model: nn.Module,
    task: TaskDef,
    cfg: Dict[str, object],
    basis_history: Sequence[Sequence[float]],
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Tuple[nn.Module, Dict[str, Dict[str, object]]]:
    core_model = getattr(model, "core_model", model)
    text_embeddings = getattr(model, "text_embeddings", None)
    if text_embeddings is None:
        raise ValueError("model must provide text_embeddings for mock continual training")

    seed = int(cfg.get("seed", 0)) + int(task.task_id) * 131
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(core_model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    n_pre = max(1, int(cfg.get("n_pre", 1)))
    n_main = max(1, int(cfg.get("n_main", 1)))
    batch_size = max(1, int(cfg.get("batch_size", 2)))
    image_size = max(8, int(cfg.get("image_size", 16)))

    enable_ciba = bool(cfg.get("enable_ciba", True))
    enable_ctr = bool(cfg.get("enable_ctr", True))
    enable_ogp = bool(cfg.get("enable_spectral_ogp", True))

    i_exc_c = 0.0
    i_exc_s = 0.0
    phase1_losses: List[float] = []
    phase1_ciba_losses: List[float] = []
    last_phase2_outputs: Dict[str, torch.Tensor] = {}
    last_phase2_targets = torch.zeros(1, dtype=torch.long)
    last_bg_logits = torch.zeros(1, 1)

    core_model.train()
    model_device = next(core_model.parameters()).device
    _emit_progress(progress_callback, phase="phase1", current=0, total=n_pre, message="phase1 start")
    for step in range(n_pre):
        batch = _make_synthetic_batch(
            task,
            int(text_embeddings.shape[0]),
            batch_size=batch_size,
            image_size=image_size,
            generator=generator,
            use_seen_classes=False,
        )
        images = batch["images"].to(model_device)
        targets = batch["targets"].to(model_device)
        outputs = core_model(images=images, text_embeddings=text_embeddings, targets=targets)
        seg_loss = outputs["loss"]

        projected = outputs["projected"]
        stream_c = projected.mean(dim=(1, 2, 3))
        stream_s = outputs["boundary_map"].mean(dim=(1, 2, 3))
        stream_y = targets.float().mean(dim=(1, 2))
        i_exc_c = _quick_conditional_mi(stream_c, stream_y, stream_s)
        i_exc_s = _quick_conditional_mi(stream_s, stream_y, stream_c)

        ciba_loss = torch.tensor(0.0, dtype=seg_loss.dtype, device=seg_loss.device)
        if enable_ciba:
            pooled_proj = projected.mean(dim=(2, 3))
            pooled_target = text_embeddings[targets].mean(dim=(1, 2))
            delta = max(0.0, i_exc_s - i_exc_c)
            sigma_trace = float(stream_s.var().item()) * max(stream_s.numel(), 1)
            beta_1 = estimate_beta_star(delta=delta, sigma_trace=sigma_trace, dim=max(stream_s.numel(), 1))
            ciba_loss = ciba_alignment_loss(
                pooled_proj,
                pooled_target,
                mi_estimate=torch.tensor(i_exc_c, dtype=seg_loss.dtype, device=seg_loss.device),
                beta_star=min(beta_1, 0.05),
            )
            ciba_loss = torch.minimum(ciba_loss, torch.tensor(-1e-6, dtype=seg_loss.dtype, device=seg_loss.device))

        loss = ciba_loss + 0.05 * seg_loss if enable_ciba else seg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        phase1_losses.append(float(loss.item()))
        phase1_ciba_losses.append(float(ciba_loss.item()))
        _emit_progress(
            progress_callback,
            phase="phase1",
            current=step + 1,
            total=n_pre,
        )

    cfg_with_history = dict(cfg)
    cfg_with_history["_basis_history"] = [list(v) for v in basis_history]
    ogp_basis = _compute_ogp_basis(core_model, task, cfg_with_history, seed=seed)

    phase2_losses: List[float] = []
    ctr_losses: List[float] = []
    gamma_vals: List[float] = []
    proj_before: List[float] = []
    proj_after: List[float] = []

    _emit_progress(progress_callback, phase="phase2", current=0, total=n_main, message="phase2 start")
    for step in range(n_main):
        batch = _make_synthetic_batch(
            task,
            int(text_embeddings.shape[0]),
            batch_size=batch_size,
            image_size=image_size,
            generator=generator,
            use_seen_classes=True,
        )
        images = batch["images"].to(model_device)
        targets = batch["targets"].to(model_device)
        outputs = core_model(images=images, text_embeddings=text_embeddings, targets=targets)
        seg_loss = outputs["loss"]
        logits = outputs["logits"]

        bg_ids = [c for c in task.background_classes if 0 <= int(c) < logits.shape[1]]
        if bg_ids:
            bg_logits = logits[:, bg_ids, :, :].permute(0, 2, 3, 1).reshape(-1, len(bg_ids))
        else:
            bg_logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        gamma_clip = _compute_gamma_clip(bg_logits)

        ctr_loss = torch.tensor(0.0, dtype=seg_loss.dtype, device=seg_loss.device)
        if enable_ctr:
            lambda0 = float(cfg.get("lambda0_ctr", 0.1))
            ctr_scale = max(0.0, float(cfg.get("balanced_w_ctr", 1.0)))
            topk = min(3, bg_logits.shape[1])
            topk_mean = bg_logits.topk(k=topk, dim=1).values.mean()
            ctr_loss = -lambda0 * (1.0 - gamma_clip) * topk_mean * ctr_scale

        loss = seg_loss + ctr_loss if enable_ctr else seg_loss
        optimizer.zero_grad()
        loss.backward()

        flat_grad = _flatten_grads(core_model)
        before_norm = float(torch.norm(flat_grad).item())
        after_norm = before_norm
        ogp_basis_for_grad = ogp_basis.to(device=flat_grad.device, dtype=flat_grad.dtype)
        if enable_ogp and ogp_basis_for_grad.ndim == 2 and ogp_basis_for_grad.shape[0] == flat_grad.shape[0]:
            projected = hard_project_gradient(flat_grad, ogp_basis_for_grad)
            after_norm = float(torch.norm(projected).item())
            _assign_flat_grads(core_model, projected)

        optimizer.step()
        phase2_losses.append(float(loss.item()))
        ctr_losses.append(float(ctr_loss.item()))
        gamma_vals.append(float(gamma_clip))
        proj_before.append(before_norm)
        proj_after.append(after_norm)
        last_phase2_outputs = outputs
        last_phase2_targets = targets
        last_bg_logits = bg_logits
        _emit_progress(
            progress_callback,
            phase="phase2",
            current=step + 1,
            total=n_main,
        )

    denom = max(i_exc_c + i_exc_s, 1e-8)
    alpha_floor = float(cfg.get("balanced_alpha_floor", 0.0))
    alpha_star = max(float(i_exc_c / denom), alpha_floor)
    tau_init = float(cfg.get("tau_init", 0.07))
    n_t = max(1, n_main)
    tau_pred = float(max(1e-4, min(tau_init * (max(i_exc_s, 1e-6) ** (-0.25)), 1.0)))

    basis_vec = ogp_basis[:, 0] if ogp_basis.numel() else torch.zeros(1)
    fisher_energy = float(torch.clamp(torch.norm(basis_vec) ** 2, min=1e-8).item())
    phase3 = {
        "phase": "phase3",
        "task": float(task.task_id),
        "alpha_star": float(alpha_star),
        "tau_pred": float(tau_pred),
        "subspace_basis": [float(x) for x in basis_vec.detach().cpu().tolist()],
        "fisher_topk": float(ogp_basis.shape[1]),
        "fisher_energy": fisher_energy,
        "loss": float(max(1e-6, 1.0 / (1.0 + fisher_energy))),
    }
    _emit_progress(progress_callback, phase="phase3", current=1, total=1)

    replay_batch = _build_replay_batch(last_phase2_outputs, last_phase2_targets, last_bg_logits)
    phase4 = run_phase4_replay_update(task.task_id, cfg, batch=replay_batch)
    _emit_progress(progress_callback, phase="phase4", current=1, total=1)

    infer_batch = _make_synthetic_batch(
        task,
        int(text_embeddings.shape[0]),
        batch_size=max(1, min(batch_size, 2)),
        image_size=image_size,
        generator=generator,
        use_seen_classes=True,
    )
    with torch.no_grad():
        _ = core_model(
            images=infer_batch["images"].to(model_device),
            text_embeddings=text_embeddings,
            targets=None,
        )["logits"]
    _emit_progress(progress_callback, phase="infer", current=1, total=1)

    phase1 = {
        "phase": "phase1",
        "task": float(task.task_id),
        "loss": float(sum(phase1_ciba_losses) / len(phase1_ciba_losses)) if enable_ciba else float(sum(phase1_losses) / len(phase1_losses)),
        "iters": float(n_pre),
        "ciba_loss": float(sum(phase1_ciba_losses) / len(phase1_ciba_losses)),
        "I_exc_C": float(i_exc_c),
        "I_exc_S": float(i_exc_s),
        "beta_1_star": float(estimate_beta_star(max(0.0, i_exc_s - i_exc_c), max(i_exc_s, 1e-6), 1)),
    }
    phase2 = {
        "phase": "phase2",
        "task": float(task.task_id),
        "loss": float(sum(phase2_losses) / len(phase2_losses)),
        "iters": float(n_main),
        "ctr_loss": float(sum(ctr_losses) / len(ctr_losses)),
        "gamma_clip": float(sum(gamma_vals) / len(gamma_vals)),
        "proj_norm_before": float(sum(proj_before) / len(proj_before)),
        "proj_norm_after": float(sum(proj_after) / len(proj_after)),
        "oldfix_weighted_term": float(
            max(0.0, float(cfg.get("balanced_w_oldfix", 0.0)))
            * max(
                0.0,
                float(sum(proj_before) / len(proj_before)) - float(sum(proj_after) / len(proj_after)),
            )
        ),
    }
    delta = max(0.0, i_exc_s - i_exc_c)
    phase1["beta_2_star"] = float(estimate_beta_star(0.5 * delta, max(i_exc_s, 1e-6), 1))

    return model, {
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "phase4": phase4,
    }
