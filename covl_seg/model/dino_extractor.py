from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


class DINOv2FeatureExtractor(nn.Module):
    """DINOv2 multi-scale feature extractor using forward hooks on transformer blocks.

    Extracts features at strides 8, 16, and 32 by capturing intermediate
    block outputs and reshaping/removing the CLS token.
    """

    _BLOCK_INDICES = {
        "dinov2_vits14": (3, 6, 11),
        "dinov2_vitb14": (3, 6, 11),
        "dinov2_vitl14": (6, 12, 23),
    }

    _TIMM_NAME_MAP = {
        "dinov2_vits14": "vit_small_patch14_dinov2_lvd142m",
        "dinov2_vitb14": "vit_base_patch14_dinov2_lvd142m",
        "dinov2_vitl14": "vit_large_patch14_dinov2_lvd142m",
    }

    PATCH_SIZE = 14
    STRIDES = {"res3": 8, "res4": 16, "res5": 32}

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        freeze: bool = True,
        model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self._intermediate: Dict[str, torch.Tensor] = {}

        if model is not None:
            self.model = model
        else:
            self.model = self._load_model(model_name)

        self._hook_indices = self._compute_hook_indices()
        self._hooks = self._register_hooks()

        if freeze:
            self._freeze()

    def _load_model(self, model_name: str) -> nn.Module:
        try:
            model = torch.hub.load("facebookresearch/dinov2", model_name)
            return model
        except Exception:
            pass
        try:
            import timm

            timm_name = self._TIMM_NAME_MAP.get(model_name, model_name)
            model = timm.create_model(timm_name, pretrained=True)
            return model
        except Exception:
            pass
        raise RuntimeError(
            f"Could not load DINOv2 model '{model_name}' via torch.hub or timm."
        )

    def _compute_hook_indices(self) -> Dict[str, int]:
        if self.model_name in self._BLOCK_INDICES:
            indices = self._BLOCK_INDICES[self.model_name]
            return {"res3": indices[0], "res4": indices[1], "res5": indices[2]}
        blocks = self._get_blocks()
        n = len(blocks)
        return {"res3": n // 4, "res4": n // 2, "res5": n - 1}

    def _get_blocks(self):
        if hasattr(self.model, "blocks"):
            return list(self.model.blocks.children())
        return list(self.model.children())

    def _register_hooks(self):
        blocks = self._get_blocks()
        hooks = []
        for name, idx in self._hook_indices.items():
            hooks.append(blocks[idx].register_forward_hook(self._make_hook(name)))
        return hooks

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            self._intermediate[name] = output

        return hook_fn

    def _freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = images.shape
        self._intermediate.clear()

        with torch.no_grad():
            _ = self.model(images)

        result = {}
        for name, stride in self.STRIDES.items():
            feat = self._intermediate[name]
            feat = feat[:, 1:, :]
            n_patches_h = H // self.PATCH_SIZE
            n_patches_w = W // self.PATCH_SIZE
            feat = feat.reshape(B, n_patches_h, n_patches_w, -1)
            feat = feat.permute(0, 3, 1, 2)
            target_h = H // stride
            target_w = W // stride
            if feat.shape[2] != target_h or feat.shape[3] != target_w:
                feat = F.interpolate(
                    feat,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            result[name] = feat
        return result