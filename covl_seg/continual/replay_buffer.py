from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ReplayItem:
    image_path: str
    label_path: str
    class_id: int
    priority: float


class SACRReplayBuffer:
    def __init__(self, max_total_items: int, max_per_class: int):
        if max_total_items <= 0:
            raise ValueError("max_total_items must be positive")
        if max_per_class <= 0:
            raise ValueError("max_per_class must be positive")
        self.max_total_items = max_total_items
        self.max_per_class = max_per_class
        self._items: List[ReplayItem] = []

    def _sorted(self) -> List[ReplayItem]:
        return sorted(
            self._items,
            key=lambda x: (x.priority, x.class_id, x.image_path, x.label_path),
            reverse=True,
        )

    def _enforce_per_class_limit(self) -> None:
        by_class: Dict[int, List[ReplayItem]] = {}
        for item in self._sorted():
            class_items = by_class.setdefault(item.class_id, [])
            if len(class_items) < self.max_per_class:
                class_items.append(item)

        new_items: List[ReplayItem] = []
        for class_items in by_class.values():
            new_items.extend(class_items)
        self._items = new_items

    def _enforce_total_limit(self) -> None:
        self._items = self._sorted()[: self.max_total_items]

    def add(self, item: ReplayItem) -> None:
        self._items.append(item)
        self._enforce_per_class_limit()
        self._enforce_total_limit()

    def items(self) -> List[ReplayItem]:
        return self._sorted()

    def save(self, file_path: Path) -> None:
        payload = {
            "max_total_items": self.max_total_items,
            "max_per_class": self.max_per_class,
            "items": [asdict(item) for item in self._sorted()],
        }
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, file_path: Path) -> "SACRReplayBuffer":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        buffer = cls(
            max_total_items=int(payload["max_total_items"]),
            max_per_class=int(payload["max_per_class"]),
        )
        for raw in payload.get("items", []):
            buffer.add(
                ReplayItem(
                    image_path=str(raw["image_path"]),
                    label_path=str(raw["label_path"]),
                    class_id=int(raw["class_id"]),
                    priority=float(raw["priority"]),
                )
            )
        return buffer
