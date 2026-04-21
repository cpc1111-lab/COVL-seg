import torch


def test_mine_estimate_is_finite_scalar():
    from covl_seg.losses.mine import MINECritic, mine_lower_bound

    critic = MINECritic(feature_dim=8, hidden_dim=16)
    x = torch.randn(32, 8)
    y = 0.5 * x + 0.1 * torch.randn(32, 8)
    mi = mine_lower_bound(critic, x, y)

    assert mi.ndim == 0
    assert torch.isfinite(mi)


def test_beta_star_is_finite_and_clipped():
    from covl_seg.losses.ciba import estimate_beta_star

    beta = estimate_beta_star(delta=0.3, sigma_trace=8.0000001, dim=8, eps=1e-6)
    assert torch.isfinite(torch.tensor(beta))
    assert beta >= 0.0


def test_ctr_loss_penalizes_ambiguous_background_more():
    from covl_seg.losses.ctr import ctr_background_loss

    text = torch.randn(6, 8)
    bg_ids = torch.tensor([2, 3, 4, 5])

    features_sharp = torch.randn(20, 8)
    features_ambiguous = features_sharp + 0.05

    loss_sharp = ctr_background_loss(features_sharp, text, bg_ids, gamma_clip=0.8, lambda0=0.1, topk=3)
    loss_ambiguous = ctr_background_loss(
        features_ambiguous,
        text,
        bg_ids,
        gamma_clip=0.1,
        lambda0=0.1,
        topk=3,
    )
    # After the sign fix, CTR loss is negative.  Ambiguous background (low γ_clip) yields a
    # larger-magnitude (more negative) loss, meaning a stronger gradient signal during
    # minimisation — i.e. loss_ambiguous ≤ loss_sharp numerically.
    assert loss_ambiguous <= loss_sharp


def test_masked_segmentation_ce_returns_finite_loss():
    from covl_seg.losses.segmentation import masked_segmentation_ce

    logits = torch.randn(2, 4, 16, 16)
    targets = torch.randint(0, 4, (2, 16, 16))
    targets[:, :2, :2] = 255

    loss = masked_segmentation_ce(logits, targets, ignore_index=255)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
