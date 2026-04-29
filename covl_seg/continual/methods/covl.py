from __future__ import annotations

from .base import MethodState


class COVLMethod:
    name = "covl"

    def __init__(self, config: dict):
        self.config = config

    def before_task(self, task_state: dict) -> None:
        return None

    def phase_overrides(self) -> dict:
        return {
            "enable_ciba": bool(self.config.get("enable_ciba", True)),
            "enable_ctr": bool(self.config.get("enable_ctr", True)),
            "enable_spectral_ogp": bool(self.config.get("enable_spectral_ogp", True)),
            "enable_sacr": bool(self.config.get("enable_sacr", True)),
            "ciba_weight": float(self.config.get("ciba_weight", 0.1)),
            "ctr_weight": float(self.config.get("ctr_weight", 0.1)),
            "ogp_topk": int(self.config.get("ogp_topk", 10)),
        }

    def after_task(self, task_state: dict) -> MethodState:
        return MethodState(values={"covl_enabled": 1.0})