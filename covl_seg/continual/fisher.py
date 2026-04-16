from typing import Callable, Tuple

import torch


def fisher_matvec_from_gradients(gradients: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Implicit Fisher matvec using per-sample gradient rows.

    Args:
        gradients: Tensor `[N, P]` where each row is `g_i`.
        vector: Tensor `[P]`.
    Returns:
        Tensor `[P]` equal to `(1/N) sum_i <g_i, v> g_i`.
    """
    if gradients.ndim != 2:
        raise ValueError("gradients must be [N, P]")
    if vector.ndim != 1:
        raise ValueError("vector must be [P]")
    if gradients.shape[1] != vector.shape[0]:
        raise ValueError("vector length must match gradients feature dim")

    coeff = gradients @ vector
    return (coeff.unsqueeze(1) * gradients).mean(dim=0)


def top_eigenvectors_power(
    matvec_fn: Callable[[torch.Tensor], torch.Tensor],
    dim: int,
    k: int,
    num_iters: int = 100,
    tol: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k eigenspace of a symmetric PSD operator with power iteration."""
    if k <= 0:
        raise ValueError("k must be positive")
    if k > dim:
        raise ValueError("k must be <= dim")

    eigenvectors = []
    eigenvalues = []

    for _ in range(k):
        vec = torch.randn(dim)
        vec = vec / (vec.norm() + 1e-12)

        for _ in range(num_iters):
            next_vec = matvec_fn(vec)

            for prev_vec in eigenvectors:
                next_vec = next_vec - prev_vec * torch.dot(prev_vec, next_vec)

            norm = next_vec.norm()
            if norm < tol:
                break
            next_vec = next_vec / norm

            if torch.norm(next_vec - vec) < tol:
                vec = next_vec
                break
            vec = next_vec

        value = torch.dot(vec, matvec_fn(vec))
        eigenvectors.append(vec)
        eigenvalues.append(value)

    vec_stack = torch.stack(eigenvectors, dim=1)
    val_stack = torch.stack(eigenvalues)
    return vec_stack, val_stack
