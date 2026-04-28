"""Continual-learning utilities for COVL-Seg."""

from .ewc import EWCRegularizer
from .fisher import fisher_matvec_from_gradients, top_eigenvectors_power
from .task_partition import TaskDef, TaskPlan, build_task_plan
from .replay_buffer import ReplayItem, SACRReplayBuffer
from .spectral_ogp import hard_project_gradient

__all__ = [
    "EWCRegularizer",
    "fisher_matvec_from_gradients",
    "hard_project_gradient",
    "ReplayItem",
    "SACRReplayBuffer",
    "TaskDef",
    "TaskPlan",
    "build_task_plan",
    "top_eigenvectors_power",
]
