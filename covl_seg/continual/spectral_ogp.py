from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


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


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    """Flatten all parameter gradients into a single 1D tensor."""
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().view(-1))
    if not grads:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(grads)


def unflatten_and_project(model: nn.Module, gradient: torch.Tensor, basis: torch.Tensor) -> None:
    """Project a flattened gradient onto the orthogonal complement of basis,
    then assign back to model parameter gradients in-place.

    Args:
        model: The model whose parameter gradients will be modified.
        gradient: Flattened gradient vector of shape `[P]`.
        basis: Orthonormal basis of shape `[P, K]`.
    """
    projected = hard_project_gradient(gradient, basis)
    offset = 0
    for param in model.parameters():
        if param.grad is not None:
            numel = param.grad.numel()
            if offset + numel <= projected.shape[0]:
                param.grad.copy_(projected[offset:offset + numel].view_as(param.grad))
            offset += numel


def compute_gradient_basis(
    model: nn.Module,
    dataloader,
    loss_fn,
    n_samples: int = 200,
    top_k: int = 10,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute an orthonormal basis from model gradients for OGP projection.

    Collects gradients from mini-batches, computes SVD, and returns the
    top-k left singular vectors as an orthonormal basis.

    Args:
        model: The model to compute gradients for.
        dataloader: Data loader producing (images, targets) batches.
        loss_fn: Callable(model_output, targets) -> scalar loss.
        n_samples: Number of gradient samples to collect.
        top_k: Number of basis vectors to return.
        device: Device for computation.

    Returns:
        Orthonormal basis tensor of shape `[P, K]` where P is total
        parameters and K = min(top_k, n_samples).
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    gradients: List[torch.Tensor] = []
    n_collected = 0

    for batch in dataloader:
        if n_collected >= n_samples:
            break
        model.zero_grad()
        images = batch["image"].to(device)
        targets = batch["sem_seg"].to(device)
        try:
            output = model(images, targets=targets)
            loss = output.get("loss", loss_fn(output, targets))
        except TypeError:
            loss = loss_fn(output if isinstance(output, torch.Tensor) else None, targets)

        if isinstance(loss, torch.Tensor) and loss.requires_grad:
            loss.backward()
            grad_vec = flatten_gradients(model)
            if grad_vec.numel() > 0:
                gradients.append(grad_vec.cpu())
            n_collected += 1
        model.zero_grad()

    if not gradients:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return torch.zeros(num_params, 1, dtype=torch.float32)

    grad_matrix = torch.stack(gradients, dim=1)
    try:
        U, S, Vh = torch.linalg.svd(grad_matrix, full_matrices=False)
    except RuntimeError:
        return torch.eye(grad_matrix.shape[0], 1, dtype=torch.float32)[:, :1]

    k = min(top_k, U.shape[1])
    return U[:, :k].to(device)