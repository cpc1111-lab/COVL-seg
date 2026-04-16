from typing import Optional

import torch
from torch import nn


class BoundaryDetector(nn.Module):
    """Boundary map estimator using Sobel gradients.

    If attention maps are provided, they are used as the source signal.
    Otherwise, boundaries are estimated from grayscale image intensity.
    """

    def __init__(self, threshold: float = 0.15):
        super().__init__()
        self.threshold = threshold
        self.register_buffer(
            "sobel_x",
            torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]]),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]]),
        )

    def _grad_mag(self, signal: torch.Tensor) -> torch.Tensor:
        grad_x = nn.functional.conv2d(signal, self.sobel_x, padding=1)
        grad_y = nn.functional.conv2d(signal, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)

    def forward(
        self,
        images: torch.Tensor,
        attention_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_map is not None:
            source = attention_map
            if source.ndim == 3:
                source = source.unsqueeze(1)
            elif source.ndim == 4 and source.shape[1] != 1:
                source = source.mean(dim=1, keepdim=True)
        else:
            source = images.mean(dim=1, keepdim=True)

        grad_mag = self._grad_mag(source)
        grad_mag = grad_mag / (grad_mag.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        return (grad_mag >= self.threshold).to(grad_mag.dtype)
