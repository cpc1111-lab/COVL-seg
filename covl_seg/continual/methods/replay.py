from __future__ import annotations

from .base import MethodState


class ReplayMethod:
    name = "replay"

    def __init__(self, config: dict):
        self.config = config
        self.replay_ratio = float(config.get("replay_ratio", 0.25))

    def before_task(self, task_state: dict) -> None:
        return None

    def phase_overrides(self) -> dict:
        return {
            "enable_ciba": False,
            "enable_ctr": False,
            "enable_sacr": True,
            "mix_ratio": [max(1, int(round((1 - self.replay_ratio) * 4))), 1],
        }

    def after_task(self, task_state: dict) -> MethodState:
        return MethodState(values={"replay_ratio": self.replay_ratio})
