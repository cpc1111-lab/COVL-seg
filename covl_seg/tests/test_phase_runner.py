from covl_seg.engine.phase_runner import (
    run_phase1_hciba,
    run_phase2_joint,
    run_phase3_subspace_and_fusion,
    run_phase4_replay_update,
)


def test_run_phase1_returns_hciba_proxy_metrics():
    batch = {
        "features_c": [0.2, 0.4, 0.6],
        "features_s": [0.1, 0.5, 0.7],
        "targets": [0.3, 0.4, 0.8],
    }
    out = run_phase1_hciba(task_id=1, cfg={"n_pre": 5}, batch=batch)
    assert "beta_1_star" in out
    assert "beta_2_star" in out
    assert "I_exc_C" in out
    assert "I_exc_S" in out
    assert out["beta_1_star"] >= 0.0
    assert out["beta_2_star"] >= 0.0


def test_run_phase2_reports_ctr_and_projection_metrics():
    out = run_phase2_joint(task_id=2, cfg={"n_main": 10, "enable_ctr": True, "enable_spectral_ogp": True})
    assert "ctr_loss" in out
    assert "gamma_clip" in out
    assert "proj_norm_before" in out
    assert "proj_norm_after" in out
    assert out["proj_norm_after"] <= out["proj_norm_before"]
    assert 0.0 <= out["gamma_clip"] <= 1.0


def test_run_phase2_applies_balanced_oldfix_weight():
    base = run_phase2_joint(task_id=3, cfg={"n_main": 10, "balanced_w_oldfix": 0.0})
    weighted = run_phase2_joint(task_id=3, cfg={"n_main": 10, "balanced_w_oldfix": 2.0})

    assert "oldfix_weighted_term" in base
    assert "oldfix_weighted_term" in weighted
    assert base["oldfix_weighted_term"] == 0.0
    assert weighted["oldfix_weighted_term"] > 0.0


def test_run_phase2_clamps_negative_balanced_w_ctr_before_scaling_ctr_loss():
    zero_scaled = run_phase2_joint(task_id=4, cfg={"n_main": 10, "enable_ctr": True, "balanced_w_ctr": 0.0})
    negative_scaled = run_phase2_joint(task_id=4, cfg={"n_main": 10, "enable_ctr": True, "balanced_w_ctr": -2.0})

    assert zero_scaled["ctr_loss"] >= 0.0
    assert negative_scaled["ctr_loss"] == zero_scaled["ctr_loss"]


def test_run_phase3_returns_subspace_and_fusion_keys():
    out = run_phase3_subspace_and_fusion(task_id=1, cfg={"eps_f": 0.05})
    assert "alpha_star" in out
    assert "tau_pred" in out
    assert "fisher_topk" in out
    assert "fisher_energy" in out
    assert out["alpha_star"] > 0.0
    assert out["tau_pred"] > 0.0
    assert out["fisher_topk"] >= 1
    assert out["fisher_energy"] > 0.0
    assert "subspace_basis" in out


def test_run_phase4_returns_replay_selection_stats():
    out = run_phase4_replay_update(task_id=2, cfg={"m_max_per_class": 10})
    assert "replay_selected" in out
    assert out["replay_selected"] >= 1


def test_run_phase4_uses_balanced_rho_in_priority_telemetry():
    base = run_phase4_replay_update(task_id=2, cfg={"m_max_per_class": 10})
    weighted = run_phase4_replay_update(
        task_id=2,
        cfg={"m_max_per_class": 10, "balanced_rho_new": 0.5, "balanced_rho_old": 0.25},
    )

    assert "replay_rho_new" in weighted
    assert "replay_rho_old" in weighted
    assert "replay_priority_new_term" in weighted
    assert "replay_priority_old_term" in weighted
    assert weighted["replay_rho_new"] == 0.5
    assert weighted["replay_rho_old"] == 0.25
    assert weighted["replay_priority_new_term"] > base["replay_priority_new_term"]
    assert weighted["replay_priority_old_term"] > base["replay_priority_old_term"]
