"""Training and evaluation engine for COVL-Seg."""

from .evaluator import compute_basic_miou, summarize_metrics
from .hooks import append_metrics_jsonl
from .open_continual_trainer import OpenContinualTrainer
from .open_vocab_eval import OpenVocabEvaluator
from .trainer import FourPhaseTrainer, Phase, PhaseController

__all__ = [
    "append_metrics_jsonl",
    "compute_basic_miou",
    "FourPhaseTrainer",
    "OpenContinualTrainer",
    "OpenVocabEvaluator",
    "Phase",
    "PhaseController",
    "summarize_metrics",
]
