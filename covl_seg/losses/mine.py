import math
from typing import Tuple

import torch
from torch import nn


class MINECritic(nn.Module):
    """Two-layer MINE critic network."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape:
            raise ValueError("x and y must share the same shape")
        return self.net(torch.cat([x, y], dim=1)).squeeze(1)


class ConditionalMINECritic(nn.Module):
    """Two-layer conditional MINE critic network."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.shape != y.shape or x.shape != cond.shape:
            raise ValueError("x, y, and cond must share the same shape")
        return self.net(torch.cat([x, y, cond], dim=1)).squeeze(1)


def mine_lower_bound(critic: MINECritic, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    joint_score = critic(x, y).mean()
    perm = torch.randperm(y.shape[0], device=y.device)
    marginal_score = critic(x, y[perm])
    log_marginal = torch.logsumexp(marginal_score, dim=0) - math.log(marginal_score.numel())
    return joint_score - log_marginal


def mine_loss(critic: MINECritic, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -mine_lower_bound(critic, x, y)


def conditional_mine_lower_bound(
    critic: ConditionalMINECritic,
    x: torch.Tensor,
    y: torch.Tensor,
    cond: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    joint_score = critic(x, y, cond).mean()
    perm = torch.randperm(x.shape[0], device=x.device)
    marginal_score = critic(x[perm], y, cond)
    log_marginal = torch.logsumexp(marginal_score, dim=0) - math.log(marginal_score.numel())
    return joint_score - log_marginal


def conditional_mine_loss(
    critic: ConditionalMINECritic,
    x: torch.Tensor,
    y: torch.Tensor,
    cond: torch.Tensor,
) -> torch.Tensor:
    return -conditional_mine_lower_bound(critic, x, y, cond)


def paired_batch(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must share batch size")
    return x, y
