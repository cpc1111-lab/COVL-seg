from enum import Enum
from typing import Dict, List

import torch
from torch import nn


class Phase(str, Enum):
    PHASE1 = "phase1"
    PHASE2 = "phase2"
    PHASE3 = "phase3"
    PHASE4 = "phase4"


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


class PhaseController:
    """Controls trainable parameter groups across the 4-phase loop."""

    def __init__(self, f_s: nn.Module, phi: nn.Module, mine: nn.Module):
        self.f_s = f_s
        self.phi = phi
        self.mine = mine
        self.current_phase = None

    def set_phase(self, phase: Phase) -> None:
        if phase == Phase.PHASE1:
            _set_requires_grad(self.f_s, False)
            _set_requires_grad(self.phi, True)
            _set_requires_grad(self.mine, True)
        elif phase == Phase.PHASE2:
            _set_requires_grad(self.f_s, True)
            _set_requires_grad(self.phi, True)
            _set_requires_grad(self.mine, False)
        elif phase in (Phase.PHASE3, Phase.PHASE4):
            _set_requires_grad(self.f_s, False)
            _set_requires_grad(self.phi, False)
            _set_requires_grad(self.mine, False)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        self.current_phase = phase


class FourPhaseTrainer:
    """Minimal four-phase trainer with deterministic integration behavior."""

    def __init__(self, model: nn.Module, controller: PhaseController, seed: int = 0):
        self.model = model
        self.controller = controller
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)

    def phase_step(self, phase: Phase, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.controller.set_phase(phase)
        phase_loss = float(torch.rand(1, generator=self.rng).item())
        if phase == Phase.PHASE1:
            return {"phase": torch.tensor(1), "loss": torch.tensor(phase_loss)}
        if phase == Phase.PHASE2:
            return {"phase": torch.tensor(2), "loss": torch.tensor(phase_loss)}
        if phase == Phase.PHASE3:
            return {"phase": torch.tensor(3), "loss": torch.tensor(phase_loss)}
        return {"phase": torch.tensor(4), "loss": torch.tensor(phase_loss)}

    def run_tasks(self, num_tasks: int, start_task: int = 0) -> List[Dict[str, float]]:
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")

        records: List[Dict[str, float]] = []
        phases = [Phase.PHASE1, Phase.PHASE2, Phase.PHASE3, Phase.PHASE4]

        for task_id in range(start_task + 1, start_task + num_tasks + 1):
            dummy_batch = {"x": torch.randn(2, 4, generator=self.rng)}
            for phase in phases:
                output = self.phase_step(phase=phase, batch=dummy_batch)
                records.append(
                    {
                        "task": float(task_id),
                        "phase": phase.value,
                        "loss": float(output["loss"].item()),
                    }
                )
        return records
