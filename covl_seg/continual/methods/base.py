from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol


@dataclass
class MethodState:
    values: Dict[str, float]


class ContinualMethod(Protocol):
    name: str

    def before_task(self, task_state: dict) -> None:
        ...

    def phase_overrides(self) -> dict:
        ...

    def after_task(self, task_state: dict) -> MethodState:
        ...
