from pathlib import Path

from .split_utils import contiguous_groups, ensure_split


def split_path() -> Path:
    groups = contiguous_groups(total=150, group_sizes=[10] * 15)
    return ensure_split("ade20k_15.json", groups)
