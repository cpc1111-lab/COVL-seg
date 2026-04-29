from __future__ import annotations

from .base import MethodState


class EWCMethod:
    name = "ewc"

    def __init__(self, config: dict):
        self.lambda_ewc = float(config.get("ewc_lambda", 5000.0))
        self.config = config

    def before_task(self, task_state: dict) -> None:
        return None

    def phase_overrides(self) -> dict:
        return {
            "enable_ewc": True,
            "ewc_lambda": self.lambda_ewc,
            "enable_ciba": False,
            "enable_ctr": False,
            "enable_spectral_ogp": False,
            "enable_sacr": False,
        }

    def after_task(self, task_state: dict) -> MethodState:
        return MethodState(values={"ewc_enabled": 1.0, "ewc_lambda": self.lambda_ewc})