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
    return lambda0 * (1.0 - gamma_clip) * topk_mean
