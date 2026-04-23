import torch

from covl_seg.losses.mine import (
    ConditionalMINECritic,
    MINECritic,
    conditional_mine_lower_bound,
    conditional_mine_loss,
    mine_lower_bound,
    mine_loss,
)


class _FixedMINECritic:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), self.value, dtype=x.dtype, device=x.device)


class _FixedConditionalMINECritic:
    def __init__(self, value: float):
        self.value = value

    def __call__(self, x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0],), self.value, dtype=x.dtype, device=x.device)


def _train_marginal_bound(x: torch.Tensor, y: torch.Tensor, n_steps: int = 120) -> float:
    critic = MINECritic(feature_dim=1, hidden_dim=32)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = mine_loss(critic, x, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(mine_lower_bound(critic, x, y).item())


def _train_conditional_bound(x: torch.Tensor, y: torch.Tensor, cond: torch.Tensor, n_steps: int = 120) -> float:
    critic = ConditionalMINECritic(feature_dim=1, hidden_dim=32)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-2)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = conditional_mine_loss(critic, x, y, cond)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(conditional_mine_lower_bound(critic, x, y, cond).item())


def test_conditional_mine_critic_returns_per_sample_scores():
    critic = ConditionalMINECritic(feature_dim=3, hidden_dim=16)
    x = torch.randn(7, 3)
    y = torch.randn(7, 3)
    cond = torch.randn(7, 3)

    scores = critic(x, y, cond)

    assert scores.shape == (7,)


def test_conditional_bound_approaches_marginal_when_condition_is_constant():
    torch.manual_seed(0)
    n = 128
    x = torch.randn(n, 1)
    y = x + 0.2 * torch.randn(n, 1)
    cond = torch.zeros(n, 1)

    marginal_lb = _train_marginal_bound(x, y)
    conditional_lb = _train_conditional_bound(x, y, cond)

    assert conditional_lb > 0.0
    assert conditional_lb >= 0.6 * marginal_lb


def test_conditional_bound_reduces_confounded_dependence():
    torch.manual_seed(1)
    n = 256
    cond = torch.randn(n, 1)
    x = cond + 0.1 * torch.randn(n, 1)
    y = cond + 0.1 * torch.randn(n, 1)

    marginal_lb = _train_marginal_bound(x, y)
    conditional_lb = _train_conditional_bound(x, y, cond)

    assert marginal_lb > 0.2
    assert conditional_lb < marginal_lb


def test_mine_lower_bound_is_finite_for_high_magnitude_scores():
    x = torch.randn(16, 1)
    y = torch.randn(16, 1)

    lb = mine_lower_bound(_FixedMINECritic(1000.0), x, y)

    assert torch.isfinite(lb)


def test_conditional_mine_lower_bound_is_finite_for_high_magnitude_scores():
    x = torch.randn(16, 1)
    y = torch.randn(16, 1)
    cond = torch.randn(16, 1)

    lb = conditional_mine_lower_bound(_FixedConditionalMINECritic(1000.0), x, y, cond)

    assert torch.isfinite(lb)
