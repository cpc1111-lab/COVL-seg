from __future__ import annotations

from .covl import COVLMethod
from .ewc import EWCMethod
from .none import NoneMethod
from .replay import ReplayMethod


def build_continual_method(name: str, config: dict):
    lowered = name.lower()
    if lowered == "covl":
        return COVLMethod(config)
    if lowered == "none":
        return NoneMethod(config)
    if lowered == "replay":
        return ReplayMethod(config)
    if lowered == "ewc":
        return EWCMethod(config)
    raise ValueError(f"Unsupported continual method: {name}")


__all__ = ["build_continual_method", "COVLMethod", "NoneMethod", "ReplayMethod", "EWCMethod"]
