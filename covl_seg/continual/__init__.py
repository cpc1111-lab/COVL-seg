"""Continual-learning utilities for COVL-Seg."""

from .fisher import fisher_matvec_from_gradients, top_eigenvectors_power
from .replay_buffer import ReplayItem, SACRReplayBuffer
from .spectral_ogp import hard_project_gradient

__all__ = [
    "fisher_matvec_from_gradients",
    "hard_project_gradient",
    "ReplayItem",
    "SACRReplayBuffer",
    "top_eigenvectors_power",
]
