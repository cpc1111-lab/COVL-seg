import torch


def test_power_iteration_matches_top_eigenspace():
    from covl_seg.continual.fisher import top_eigenvectors_power

    torch.manual_seed(0)
    n = 40
    a = torch.randn(n, n)
    cov = a.T @ a + 0.1 * torch.eye(n)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return cov @ v

    vecs, vals = top_eigenvectors_power(matvec_fn=matvec, dim=n, k=3, num_iters=120)

    eigvals, _ = torch.linalg.eigh(cov)
    target = eigvals.flip(0)[:3]

    assert vecs.shape == (n, 3)
    assert vals.shape == (3,)
    assert torch.allclose(vals, target, rtol=1e-2, atol=1e-2)
