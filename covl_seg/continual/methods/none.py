from __future__ import annotations

from .base import MethodState


class NoneMethod:
    name = "none"

    def __init__(self, config: dict):
        self.config = config

    def before_task(self, task_state: dict) -> None:
        return None

    def phase_overrides(self) -> dict:
        return {"enable_ciba": False, "enable_ctr": False, "enable_sacr": False}

    def after_task(self, task_state: dict) -> MethodState:
        return MethodState(values={"regularization": 0.0})
