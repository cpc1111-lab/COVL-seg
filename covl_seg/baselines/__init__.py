"""Baseline modes for COVL-Seg experiments."""

from typing import Callable, Dict

from .common import BaselineConfig
from .ewc import build as build_ewc
from .ft import build as build_ft
from .mib import build as build_mib
from .oracle import build as build_oracle
from .plop import build as build_plop
from .zscl import build as build_zscl


BASELINE_REGISTRY: Dict[str, Callable[[], BaselineConfig]] = {
    "ft": build_ft,
    "oracle": build_oracle,
    "ewc": build_ewc,
    "mib": build_mib,
    "plop": build_plop,
    "zscl": build_zscl,
}


def resolve_baseline(name: str) -> BaselineConfig:
    key = name.lower()
    if key not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline: {name}")
    return BASELINE_REGISTRY[key]()


__all__ = [
    "BASELINE_REGISTRY",
    "BaselineConfig",
    "resolve_baseline",
]
