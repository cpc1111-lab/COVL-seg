from pathlib import Path
from typing import Dict, List, Optional

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


COCO_STUFF_164_CLASSES = [
    "unlabeled", "aquarium", "banner", "bed", "blanket", "bridge", "building-other",
    "bush", "cabinet", "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile",
    "cloth", "clothes", "clouds", "counter", "cupboard", "curtain", "cushion",
    "dirt", "door-stuff", "fence", "floor-marble", "floor-other", "floor-stone",
    "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", "furniture-other",
    "glass", "gravel", "ground-other", "hill", "house", "ice", "keyboard", "light",
    "mat", "moss", "mountain", "mud", "napkin", "net", "paper", "pavement",
    "pillow", "plant-other", "plastic", "platform", "playingfield", "rug",
    "sand", "sea", "shelf", "sky-other", "sky-weather", "snow", "solid-other",
    "spinel", "stairs", "stone", "straw", "structural-other", "tablecloth",
    "tent", "textile-other", "towel", "tree", "wall-brick", "wall-other",
    "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other", "waterdrops",
    "window-blind", "window-other", "wood", "awning", "banner", "bed", "bench",
    "bike", "bird", "boat", "book", "bottle", "bowl", "bus", "cab", "cake",
    "car", "carpet", "cat", "cell phone", "chair", "clock", "cloud", "counter",
    "cow", "cup", "curtain", "dog", "door", "fence", "floor", "flower", "food",
    "fruit", "girl", "grass", "guitar", "handle", "house", "kite", "lamp",
    "leaves", "mirror", "monkey", "mountain", "mouse", "napkin", "orange",
    "person", "plant", "plate", "poster", "road", "rock", "roof", "sheep",
    "shelf", "sign", "sky", "snow", "sofa", "stone", "stove", "streetlight",
    "sun", "table", "tower", "train", "truck", "tv", "umbrella", "wall", "window",
    "wood", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


class COCOStuffDataset(Dataset):
    """COCO-Stuff-164 semantic segmentation dataset for continual learning.

    Expected directory structure:
        root/
            images/
                train2017/
                    000000000009.jpg
                    ...
                val2017/
                    000000000139.jpg
                    ...
            annotations/
                train2017/
                    000000000009.png
                    ...
                val2017/
                    000000000139.png
                    ...
    """

    SPLIT_MAP = {"training": "train2017", "validation": "val2017"}

    def __init__(
        self,
        root: str,
        split: str = "training",
        class_names: Optional[List[str]] = None,
        transform=None,
        num_classes: int = 164,
    ):
        self.root = Path(root)
        self.split = split
        self.class_names = class_names or COCO_STUFF_164_CLASSES[:num_classes]
        self.transform = transform
        self.num_classes = num_classes

        subdir = self.SPLIT_MAP.get(split, split)
        images_dir = self.root / "images" / subdir
        masks_dir = self.root / "annotations" / subdir

        self._samples: List[Dict[str, object]] = []
        exts = ("*.jpg", "*.jpeg", "*.png")
        image_paths = sorted(
            p for ext in exts for p in images_dir.glob(ext)
        )
        for img_path in image_paths:
            mask_path = masks_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                self._samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "image_id": img_path.stem,
                })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        info = self._samples[idx]

        image = Image.open(info["image_path"]).convert("RGB")
        mask = Image.open(info["mask_path"])

        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        sem_seg = torch.from_numpy(np.array(mask)).long()
        sem_seg = sem_seg.clamp(0, self.num_classes - 1)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return {
            "image": image_tensor,
            "sem_seg": sem_seg,
            "class_names": self.class_names,
            "image_id": info["image_id"],
        }