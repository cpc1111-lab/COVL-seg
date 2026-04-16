from pathlib import Path

from .split_utils import contiguous_groups, ensure_split


def split_path() -> Path:
    group_sizes = [17, 17, 17, 17, 16, 16, 16, 16, 16, 16]
    groups = contiguous_groups(total=164, group_sizes=group_sizes)
    return ensure_split("coco_stuff_164_10.json", groups)
