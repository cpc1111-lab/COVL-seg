def test_baseline_registry_contains_all_expected_modes():
    from covl_seg.baselines import BASELINE_REGISTRY

    expected = {"ft", "oracle", "ewc", "mib", "plop", "zscl"}
    assert expected.issubset(set(BASELINE_REGISTRY.keys()))


def test_ft_disables_all_special_modules():
    from covl_seg.baselines import resolve_baseline

    cfg = resolve_baseline("ft")
    assert cfg.use_hciba is False
    assert cfg.use_spectral_ogp is False
    assert cfg.use_ctr is False
    assert cfg.use_replay is False


def test_oracle_enables_joint_training_flag():
    from covl_seg.baselines import resolve_baseline

    cfg = resolve_baseline("oracle")
    assert cfg.joint_training is True


def test_unknown_baseline_raises():
    from covl_seg.baselines import resolve_baseline

    try:
        resolve_baseline("unknown")
    except ValueError as exc:
        assert "Unknown baseline" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown baseline")
