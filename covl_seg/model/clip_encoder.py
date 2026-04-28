from typing import List, Optional

import torch
from torch import nn


class CLIPVisualEncoder(nn.Module):
    """CLIP visual encoder with hook-based intermediate feature extraction."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        clip_finetune: str = "none",
        hook_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        import open_clip

        self._model_name = model_name
        self._pretrained = pretrained
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.clip_model = model
        self.visual = model.visual

        self.dim = self.visual.transformer.width
        self.intermediate_features: List[torch.Tensor] = []

        if hook_layers is None:
            hook_layers = [3, 7]
        self.hook_layers = hook_layers
        self._hooks = self._register_hooks()
        self._apply_finetune_mode(clip_finetune)

    def _register_hooks(self):
        hooks = []
        resblocks = self.visual.transformer.resblocks
        for idx in self.hook_layers:
            if idx < len(resblocks):
                hooks.append(resblocks[idx].register_forward_hook(self._hook_fn))
        return hooks

    def _hook_fn(self, module, input, output):
        self.intermediate_features.append(output)

    def _apply_finetune_mode(self, mode: str):
        for param in self.clip_model.parameters():
            param.requires_grad = False

        if mode == "full":
            for param in self.clip_model.parameters():
                param.requires_grad = True
        elif mode == "attention":
            for name, param in self.clip_model.named_parameters():
                if "attn" in name:
                    param.requires_grad = True
        elif mode == "prompt":
            pass
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown clip_finetune mode: {mode}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.intermediate_features.clear()
        x = self.visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)],
            dim=1,
        ) + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = self.visual.transformer(x)
        x = self.visual.ln_post(x)
        return x

    def get_dense_features(self, images: torch.Tensor) -> torch.Tensor:
        self.intermediate_features.clear()
        with torch.no_grad():
            x = self.forward(images)
        if len(self.intermediate_features) > 0:
            return torch.cat(self.intermediate_features, dim=1)
        return x


class CLIPTextEncoder(nn.Module):
    """CLIP text encoder producing learnable embeddings from class names."""

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.clip_model = model
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.text_dim = model.transformer.width
        if output_dim is None:
            output_dim = model.visual.transformer.width
        self.output_dim = output_dim
        self.text_projection_layer = nn.Linear(self.text_dim, output_dim, bias=False)

        for param in self.clip_model.parameters():
            param.requires_grad = True

    def forward(self, class_names: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(class_names)
        tokens = tokens.to(next(self.parameters()).device)
        text_features = self.clip_model.encode_text(tokens)
        projected = self.text_projection_layer(text_features)
        return projected