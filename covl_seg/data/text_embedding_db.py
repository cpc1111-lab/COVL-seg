from typing import List

import torch

from covl_seg.model.clip_encoder import CLIPTextEncoder


class TextEmbeddingDB:
    def __init__(self, clip_model_name: str = "ViT-B-16"):
        self.encoder = CLIPTextEncoder(model_name=clip_model_name)
        self._cache: dict = {}

    def encode(self, class_names: List[str]) -> torch.Tensor:
        return self.encoder(class_names)

    def encode_with_cache(self, class_names: List[str]) -> torch.Tensor:
        key = tuple(class_names)
        if key in self._cache:
            return self._cache[key]
        embedding = self.encode(class_names).detach().cpu()
        self._cache[key] = embedding
        return embedding

    def clear_cache(self):
        self._cache.clear()