from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ReplayItem:
    image_path: str
    label_path: str
    class_id: int
    priority: float


class SACRReplayBuffer(Dataset):
    """Priority-based replay buffer with per-class limits for continual learning.

    Stores paths to images and labels, supports sampling for mixed training
    batches, and enforces memory limits via priority-based eviction.
    """

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

    def add_batch(self, items: List[ReplayItem]) -> None:
        for item in items:
            self._items.append(item)
        self._enforce_per_class_limit()
        self._enforce_total_limit()

    def items(self) -> List[ReplayItem]:
        return self._sorted()

    def sample(self, n: int, class_ids: Optional[List[int]] = None) -> List[ReplayItem]:
        """Sample n items uniformly, optionally restricted to specific classes."""
        if not self._items:
            return []
        pool = self._items
        if class_ids is not None:
            class_set = set(class_ids)
            pool = [item for item in pool if item.class_id in class_set]
        if not pool:
            return []
        n = min(n, len(pool))
        return random.sample(pool, n)

    def sample_paths(self, n: int, class_ids: Optional[List[int]] = None) -> Tuple[List[str], List[str], List[int]]:
        """Sample n items and return (image_paths, label_paths, class_ids)."""
        items = self.sample(n, class_ids)
        if not items:
            return [], [], []
        image_paths = [it.image_path for it in items]
        label_paths = [it.label_path for it in items]
        class_ids_out = [it.class_id for it in items]
        return image_paths, label_paths, class_ids_out

    def class_distribution(self) -> Dict[int, int]:
        """Return a count of items per class_id."""
        dist: Dict[int, int] = {}
        for item in self._items:
            dist[item.class_id] = dist.get(item.class_id, 0) + 1
        return dist

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> ReplayItem:
        return self._items[idx]

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
