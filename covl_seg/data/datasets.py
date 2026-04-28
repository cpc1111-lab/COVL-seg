from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ADE20KDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training",
        class_names: Optional[List[str]] = None,
        transform=None,
    ):
        self.root = Path(root)
        self.split = split
        self.class_names = class_names
        self.transform = transform

        images_dir = self.root / "images" / self.split
        masks_dir = self.root / "annotations" / self.split

        self._samples: List[dict] = []
        for img_path in sorted(images_dir.glob("*.jpg")):
            mask_path = masks_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                self._samples.append(
                    {"image_path": img_path, "mask_path": mask_path, "image_id": img_path.stem}
                )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        info = self._samples[idx]

        image = Image.open(info["image_path"]).convert("RGB")
        mask = Image.open(info["mask_path"])

        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        sem_seg = torch.from_numpy(np.array(mask)).long()

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return {
            "image": image_tensor,
            "sem_seg": sem_seg,
            "class_names": self.class_names,
            "image_id": info["image_id"],
        }