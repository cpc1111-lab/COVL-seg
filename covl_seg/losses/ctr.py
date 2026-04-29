import torch
from torch import nn


def ctr_background_loss(
    pixel_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    background_ids: torch.Tensor,
    gamma_clip: float,
    lambda0: float,
    topk: int = 3,
) -> torch.Tensor:
    if pixel_features.ndim != 2:
        raise ValueError("pixel_features must be [N, D]")
    text_bg = text_embeddings[background_ids]
    pixel = nn.functional.normalize(pixel_features, dim=1)
    text = nn.functional.normalize(text_bg, dim=1)
    sim = pixel @ text.T
    k = min(topk, sim.shape[1])
    topk_mean = sim.topk(k=k, dim=1).values.mean()
    return -lambda0 * (1.0 - gamma_clip) * topk_mean


def contrastive_background_loss(
    logits: torch.Tensor,
    seen_class_ids: list,
    unseen_class_ids: list,
    gamma: float = 0.5,
    lambda0: float = 0.1,
    topk: int = 3,
) -> torch.Tensor:
    """Compute contrastive loss pushing background pixels away from unseen classes.

    Operates directly on logits class probabilities. For pixels likely belonging
    to unseen classes, pushes them towards seen (background) class logits.

    Args:
        logits: [B, C, H, W] segmentation logits (C = number of classes in logits).
        seen_class_ids: List of class indices visible in current task (0-indexed into logits).
        unseen_class_ids: List of class indices not yet seen (0-indexed into logits).
        gamma: Weight balancing repulsion from unseen and attraction to seen.
        lambda0: Overall loss weight.
        topk: Number of top probabilities to average.

    Returns:
        Scalar contrastive background loss.
    """
    if not unseen_class_ids or not seen_class_ids:
        return torch.tensor(0.0, device=logits.device)

    num_classes = logits.shape[1]
    safe_unseen = [c for c in unseen_class_ids if 0 <= c < num_classes]
    safe_seen = [c for c in seen_class_ids if 0 <= c < num_classes]

    if not safe_unseen or not safe_seen:
        return torch.tensor(0.0, device=logits.device)

    unseen_tensor = torch.tensor(safe_unseen, dtype=torch.long, device=logits.device)
    seen_tensor = torch.tensor(safe_seen, dtype=torch.long, device=logits.device)

    probs = torch.softmax(logits, dim=1)

    unseen_probs = probs[:, unseen_tensor, :, :]
    bg_mask = unseen_probs.sum(dim=1) < 0.5
    if bg_mask.sum() < 10:
        bg_mask = unseen_probs.sum(dim=1) < 0.8

    num_bg = bg_mask.sum().item()
    if num_bg < 1:
        return torch.tensor(0.0, device=logits.device)

    seen_probs = probs[:, seen_tensor, :, :]
    k_unseen = min(topk, len(safe_unseen))
    k_seen = min(topk, len(safe_seen))

    unseen_topk = unseen_probs.topk(k=k_unseen, dim=1).values.mean()
    seen_bg_indices = safe_seen[:min(len(safe_seen), max(1, len(safe_seen) // 4))]
    seen_bg_tensor = torch.tensor(seen_bg_indices, dtype=torch.long, device=logits.device)
    seen_bg_probs = probs[:, seen_bg_tensor, :, :]
    k_bg = min(topk, seen_bg_probs.shape[1])
    seen_bg_topk = seen_bg_probs.topk(k=k_bg, dim=1).values.mean()

    repulsion = gamma * unseen_topk
    attraction = (1.0 - gamma) * seen_bg_topk

    loss_value = lambda0 * (repulsion - attraction)
    if torch.isnan(loss_value) or torch.isinf(loss_value):
        return torch.tensor(0.0, device=logits.device, requires_grad=False)
    return loss_value