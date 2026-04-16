from pathlib import Path

from .split_utils import contiguous_groups, ensure_split


def split_path() -> Path:
    group_sizes = [2] * 50 + [1] * 50
    groups = contiguous_groups(total=150, group_sizes=group_sizes)
    return ensure_split("ade20k_100.json", groups)
