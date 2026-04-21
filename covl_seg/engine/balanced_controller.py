from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class BalancedControllerConfig:
    epsilon_ov: float = 0.0
    target_delta_new: float = 0.0
    alpha_floor_step: float = 0.01
    alpha_floor_max: float = 1.0
    g_stab_step: float = 0.1
    g_stab_max: float = 10.0
    rho_old_step: float = 0.1
    rho_old_max: float = 10.0
    rho_new_step: float = 0.1
    rho_new_max: float = 10.0
    w_ctr_step: float = 0.05
    w_ctr_max: float = 10.0


@dataclass
class BalancedControllerState:
    alpha_floor: float = 0.0
    g_stab: float = 0.0
    rho_old: float = 0.0
    rho_new: float = 0.0
    w_ctr: float = 0.0
    ov_guard_triggered: bool = False

    @property
    def ov_guard_state(self) -> bool:
        return self.ov_guard_triggered


def _increase_with_cap(value: float, step: float, cap: float) -> float:
    return min(value + step, cap)


def update_controller_state(
    state: BalancedControllerState,
    cfg: BalancedControllerConfig,
    signals: Mapping[str, object],
) -> BalancedControllerState:
    ov_min_delta = float(signals.get("ov_min_delta", 0.0))
    old_constraint_violated = bool(signals.get("old_constraint_violated", False))
    all_constraint_violated = bool(signals.get("all_constraint_violated", False))
    delta_new = float(signals.get("delta_new", 0.0))

    next_state = BalancedControllerState(
        alpha_floor=state.alpha_floor,
        g_stab=state.g_stab,
        rho_old=state.rho_old,
        rho_new=state.rho_new,
        w_ctr=state.w_ctr,
        ov_guard_triggered=False,
    )

    if ov_min_delta < -cfg.epsilon_ov:
        next_state.ov_guard_triggered = True
        next_state.alpha_floor = _increase_with_cap(
            next_state.alpha_floor,
            cfg.alpha_floor_step,
            cfg.alpha_floor_max,
        )
        next_state.g_stab = _increase_with_cap(
            next_state.g_stab,
            cfg.g_stab_step,
            cfg.g_stab_max,
        )
        next_state.rho_old = _increase_with_cap(
            next_state.rho_old,
            cfg.rho_old_step,
            cfg.rho_old_max,
        )
    elif old_constraint_violated or all_constraint_violated:
        next_state.g_stab = _increase_with_cap(
            next_state.g_stab,
            cfg.g_stab_step,
            cfg.g_stab_max,
        )
        next_state.rho_old = _increase_with_cap(
            next_state.rho_old,
            cfg.rho_old_step,
            cfg.rho_old_max,
        )
    elif delta_new < cfg.target_delta_new:
        next_state.rho_new = _increase_with_cap(
            next_state.rho_new,
            cfg.rho_new_step,
            cfg.rho_new_max,
        )
        next_state.w_ctr = _increase_with_cap(
            next_state.w_ctr,
            cfg.w_ctr_step,
            cfg.w_ctr_max,
        )

    return next_state
