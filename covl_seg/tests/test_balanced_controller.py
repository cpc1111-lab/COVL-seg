import pytest

from covl_seg.engine.balanced_controller import (
    BalancedControllerConfig,
    BalancedControllerState,
    update_controller_state,
)


def test_new_pressure_increases_when_stable_and_delta_new_under_target():
    cfg = BalancedControllerConfig(target_delta_new=0.05, rho_new_step=0.2, w_ctr_step=0.1)
    state = BalancedControllerState(rho_new=0.3, w_ctr=0.2)

    next_state = update_controller_state(
        state,
        cfg,
        {
            "ov_min_delta": 0.01,
            "old_constraint_violated": False,
            "all_constraint_violated": False,
            "delta_new": 0.01,
        },
    )

    assert next_state.rho_new == pytest.approx(0.5)
    assert next_state.w_ctr == pytest.approx(0.3)
    assert next_state.ov_guard_triggered is False


def test_ov_guard_activates_and_alpha_floor_increases_on_ov_violation():
    cfg = BalancedControllerConfig(
        epsilon_ov=0.01,
        alpha_floor_step=0.05,
        g_stab_step=0.2,
        rho_old_step=0.1,
    )
    state = BalancedControllerState(alpha_floor=0.1, g_stab=1.0, rho_old=0.4)

    next_state = update_controller_state(
        state,
        cfg,
        {
            "ov_min_delta": -0.02,
            "old_constraint_violated": False,
            "all_constraint_violated": False,
            "delta_new": 0.5,
        },
    )

    assert next_state.ov_guard_triggered is True
    assert next_state.alpha_floor == pytest.approx(0.15)
    assert next_state.g_stab == pytest.approx(1.2)
    assert next_state.rho_old == pytest.approx(0.5)


def test_old_or_all_constraint_violation_increases_stability_and_old_pressure():
    cfg = BalancedControllerConfig(g_stab_step=0.3, rho_old_step=0.2)
    state = BalancedControllerState(g_stab=0.6, rho_old=0.4)

    next_state = update_controller_state(
        state,
        cfg,
        {
            "ov_min_delta": 0.02,
            "old_constraint_violated": True,
            "all_constraint_violated": False,
            "delta_new": 0.5,
        },
    )

    assert next_state.g_stab == pytest.approx(0.9)
    assert next_state.rho_old == pytest.approx(0.6)
    assert next_state.ov_guard_triggered is False


def test_values_are_capped_at_configured_maxima():
    cfg = BalancedControllerConfig(
        epsilon_ov=0.0,
        alpha_floor_step=0.2,
        alpha_floor_max=0.5,
        g_stab_step=0.5,
        g_stab_max=1.0,
        rho_old_step=0.5,
        rho_old_max=0.9,
    )
    state = BalancedControllerState(alpha_floor=0.45, g_stab=0.8, rho_old=0.8)

    next_state = update_controller_state(
        state,
        cfg,
        {
            "ov_min_delta": -0.1,
            "old_constraint_violated": False,
            "all_constraint_violated": False,
            "delta_new": 0.5,
        },
    )

    assert next_state.alpha_floor == 0.5
    assert next_state.g_stab == 1.0
    assert next_state.rho_old == 0.9


def test_ov_guard_not_triggered_at_exact_negative_epsilon_boundary():
    cfg = BalancedControllerConfig(epsilon_ov=0.01, target_delta_new=-1.0)
    state = BalancedControllerState(alpha_floor=0.2, g_stab=0.4, rho_old=0.6)

    next_state = update_controller_state(
        state,
        cfg,
        {
            "ov_min_delta": -0.01,
            "old_constraint_violated": False,
            "all_constraint_violated": False,
            "delta_new": 0.0,
        },
    )

    assert next_state.ov_guard_triggered is False
    assert next_state.alpha_floor == pytest.approx(0.2)
    assert next_state.g_stab == pytest.approx(0.4)
    assert next_state.rho_old == pytest.approx(0.6)


def test_new_pressure_not_increased_at_exact_target_boundary():
    cfg = BalancedControllerConfig(target_delta_new=0.2, rho_new_step=0.3, w_ctr_step=0.4)
    state = BalancedControllerState(rho_new=0.5, w_ctr=0.7)

    next_state = update_controller_state(
        state,
        cfg,
        {
            "ov_min_delta": 0.0,
            "old_constraint_violated": False,
            "all_constraint_violated": False,
            "delta_new": 0.2,
        },
    )

    assert next_state.rho_new == pytest.approx(0.5)
    assert next_state.w_ctr == pytest.approx(0.7)
    assert next_state.ov_guard_triggered is False


def test_update_controller_state_does_not_mutate_input_state():
    cfg = BalancedControllerConfig(epsilon_ov=0.01, alpha_floor_step=0.05)
    state = BalancedControllerState(
        alpha_floor=0.2,
        g_stab=0.4,
        rho_old=0.6,
        rho_new=0.8,
        w_ctr=1.0,
        ov_guard_triggered=False,
    )

    next_state = update_controller_state(
        state,
        cfg,
        {
            "ov_min_delta": -0.02,
            "old_constraint_violated": False,
            "all_constraint_violated": False,
            "delta_new": 0.0,
        },
    )

    assert next_state is not state
    assert state.alpha_floor == pytest.approx(0.2)
    assert state.g_stab == pytest.approx(0.4)
    assert state.rho_old == pytest.approx(0.6)
    assert state.rho_new == pytest.approx(0.8)
    assert state.w_ctr == pytest.approx(1.0)
    assert state.ov_guard_triggered is False


def test_ov_violation_takes_precedence_over_old_and_all_violations():
    cfg = BalancedControllerConfig(
        epsilon_ov=0.01,
        alpha_floor_step=0.05,
        g_stab_step=0.2,
        rho_old_step=0.1,
        target_delta_new=0.2,
        rho_new_step=0.3,
        w_ctr_step=0.4,
    )
    state = BalancedControllerState(
        alpha_floor=0.1,
        g_stab=0.5,
        rho_old=0.3,
        rho_new=0.7,
        w_ctr=0.9,
    )

    next_state = update_controller_state(
        state,
        cfg,
        {
            "ov_min_delta": -0.02,
            "old_constraint_violated": True,
            "all_constraint_violated": True,
            "delta_new": 0.0,
        },
    )

    assert next_state.ov_guard_triggered is True
    assert next_state.alpha_floor == pytest.approx(0.15)
    assert next_state.g_stab == pytest.approx(0.7)
    assert next_state.rho_old == pytest.approx(0.4)
    assert next_state.rho_new == pytest.approx(0.7)
    assert next_state.w_ctr == pytest.approx(0.9)
