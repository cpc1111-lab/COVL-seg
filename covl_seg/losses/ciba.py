import torch


def estimate_beta_star(delta: float, sigma_trace: float, dim: int, eps: float = 1e-6) -> float:
    denom = max((sigma_trace / max(dim, 1)) - 1.0, eps)
    beta = delta / denom
    return max(beta, 0.0)


def ciba_alignment_loss(
    projected: torch.Tensor,
    target: torch.Tensor,
    mi_estimate: torch.Tensor,
    beta_star: float,
) -> torch.Tensor:
    mse = torch.mean((projected - target) ** 2)
    return -mi_estimate + beta_star * mse
