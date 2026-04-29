import torch


def estimate_beta_star(delta: float, sigma_trace: float, dim: int, eps: float = 1e-6) -> float:
    effective_sigma = sigma_trace / max(dim, 1)
    denom = max(effective_sigma - 1.0, eps)
    if abs(denom) < 0.01:
        denom = 0.01 if denom >= 0 else -0.01
    beta = delta / denom
    return max(min(beta, 10.0), 0.0)


def ciba_alignment_loss(
    projected: torch.Tensor,
    target: torch.Tensor,
    mi_estimate: torch.Tensor,
    beta_star: float,
) -> torch.Tensor:
    mse = torch.mean((projected - target) ** 2)
    return -mi_estimate + beta_star * mse
