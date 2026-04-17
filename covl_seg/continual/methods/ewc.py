from __future__ import annotations

from .base import MethodState


class EWCMethod:
    name = "ewc"

    def __init__(self, config: dict):
        self.config = config
        self.lambda_ewc = float(config.get("ewc_lambda", 10.0))

    def before_task(self, task_state: dict) -> None:
        return None

    def phase_overrides(self) -> dict:
        return {
            "enable_ciba": False,
            "enable_ctr": False,
            "enable_sacr": False,
            "ewc_lambda": self.lambda_ewc,
        }

    def after_task(self, task_state: dict) -> MethodState:
        return MethodState(values={"ewc_lambda": self.lambda_ewc})
