import torch


def test_hard_projection_removes_subspace_component():
    from covl_seg.continual.spectral_ogp import hard_project_gradient

    torch.manual_seed(0)
    basis_raw = torch.randn(32, 5)
    basis, _ = torch.linalg.qr(basis_raw)
    grad = torch.randn(32)

    projected = hard_project_gradient(grad, basis)
    leakage = torch.norm(basis.T @ projected)

    assert leakage < 1e-5
