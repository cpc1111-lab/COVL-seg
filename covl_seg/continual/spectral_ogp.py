import torch


def hard_project_gradient(gradient: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project gradient to the orthogonal complement of `basis` columns.

    Args:
        gradient: Shape `[P]`.
        basis: Shape `[P, K]` where columns are approximately orthonormal.
    """
    if gradient.ndim != 1:
        raise ValueError("gradient must be a 1D vector")
    if basis.ndim != 2:
        raise ValueError("basis must be a 2D matrix")
    if basis.shape[0] != gradient.shape[0]:
        raise ValueError("basis first dimension must match gradient length")

    component = basis @ (basis.T @ gradient)
    return gradient - component
