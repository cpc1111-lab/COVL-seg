import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class SegmentationAugmentation:
    """Joint image+mask augmentation for segmentation tasks."""

    def __init__(self, image_size: int = 518, flip_prob: float = 0.5, scale_range=(0.5, 2.0), mean=None, std=None):
        self.image_size = image_size
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple:
        _, h, w = image.shape

        if random.random() < self.flip_prob:
            image = image.flip(-1)
            mask = mask.flip(-1)

        scale = random.uniform(*self.scale_range)
        new_h, new_w = int(h * scale), int(w * scale)
        image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(new_h, new_w), mode="nearest").squeeze(0).squeeze(0).long()

        _, ch, cw = image.shape
        th, tw = self.image_size, self.image_size
        if ch < th or cw < tw:
            image = F.interpolate(image.unsqueeze(0), size=(max(ch, th), max(cw, tw)), mode="bilinear", align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(max(ch, th), max(cw, tw)), mode="nearest").squeeze(0).squeeze(0).long()
            _, ch, cw = image.shape

        if ch > th or cw > tw:
            top = random.randint(0, ch - th) if ch > th else 0
            left = random.randint(0, cw - tw) if cw > tw else 0
            image = image[:, top:top + th, left:left + tw]
            mask = mask[top:top + th, left:left + tw]

        mean = torch.tensor(self.mean, dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=image.dtype).view(3, 1, 1)
        image = (image - mean) / std

        return image, mask


class SegmentationEvalTransform:
    """Deterministic eval-time resize + normalize."""

    def __init__(self, image_size: int = 518, mean=None, std=None):
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple:
        _, h, w = image.shape
        scale = min(self.image_size / h, self.image_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(new_h, new_w), mode="nearest").squeeze(0).squeeze(0).long()

        th, tw = self.image_size, self.image_size
        _, ch, cw = image.shape
        if ch < th or cw < tw:
            image = F.interpolate(image.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(th, tw), mode="nearest").squeeze(0).squeeze(0).long()
        elif ch != th or cw != tw:
            top = (ch - th) // 2
            left = (cw - tw) // 2
            image = image[:, top:top + th, left:left + tw]
            mask = mask[top:top + th, left:left + tw]

        mean = torch.tensor(self.mean, dtype=image.dtype).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=image.dtype).view(3, 1, 1)
        image = (image - mean) / std
        return image, mask


class ADE20KDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training",
        class_names: Optional[List[str]] = None,
        transform=None,
        augmentation=None,
        num_classes: int = 150,
        visible_class_ids: Optional[List[int]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.class_names = class_names
        self.transform = transform
        self.augmentation = augmentation
        self.num_classes = num_classes
        self.visible_class_ids = set(visible_class_ids) if visible_class_ids is not None else None

        images_dir = self.root / "images" / self.split
        masks_dir = self.root / "annotations" / self.split

        if not images_dir.exists():
            from covl_seg.data.download import ensure_ade20k
            try:
                ensure_ade20k(str(self.root))
            except Exception as exc:
                raise FileNotFoundError(
                    f"ADE20K images directory not found: {images_dir}\n"
                    f"Auto-download failed: {exc}\n"
                    f"Run: python -m covl_seg.data.download --dataset ade20k"
                ) from exc
            if not images_dir.exists():
                raise FileNotFoundError(
                    f"ADE20K images directory not found: {images_dir}\n"
                    f"Run: python -m covl_seg.data.download --dataset ade20k"
                )

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
        if sem_seg.max() >= self.num_classes:
            sem_seg[sem_seg >= self.num_classes] = 255

        if self.visible_class_ids is not None:
            valid_mask = torch.zeros_like(sem_seg, dtype=torch.bool)
            for cid in self.visible_class_ids:
                valid_mask |= (sem_seg == cid)
            sem_seg[~valid_mask & (sem_seg != 255)] = 255

        if self.augmentation is not None:
            image_tensor, sem_seg = self.augmentation(image_tensor, sem_seg)
        elif self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return {
            "image": image_tensor,
            "sem_seg": sem_seg,
            "class_names": self.class_names,
            "image_id": info["image_id"],
        }


class ClassFilteredDataset(Dataset):
    """Wraps a segmentation dataset to only include samples containing visible classes.

    Efficient filtering: reads only PNG mask files (not JPEG images) to check
    pixel class presence, avoiding full image loading.
    """

    def __init__(self, base_dataset: Dataset, visible_class_ids: List[int], min_visible_ratio: float = 0.01):
        self.base_dataset = base_dataset
        self.visible_class_ids = set(visible_class_ids)
        self.min_visible_ratio = min_visible_ratio
        self._indices: Optional[List[int]] = None

    def _filter_indices(self) -> List[int]:
        indices: List[int] = []
        n = len(self.base_dataset)
        visible_set = self.visible_class_ids | {255}
        print(f"[ClassFilteredDataset] scanning {n} mask files for {len(self.visible_class_ids)} visible classes ...", flush=True)

        # Try efficient path: read mask files directly from base_dataset._samples
        samples = getattr(self.base_dataset, '_samples', None)
        # Only use efficient path for real dataset classes with _samples containing mask_path
        has_mask_path = False
        if samples is not None and len(samples) > 0:
            has_mask_path = isinstance(samples[0], dict) and "mask_path" in samples[0]
        if has_mask_path:
            for i, sample_info in enumerate(samples):
                mask_path = sample_info.get("mask_path")
                if mask_path is None:
                    continue
                try:
                    # Read PNG header only to get unique pixel values quickly
                    import numpy as np
                    from PIL import Image
                    mask = np.array(Image.open(mask_path))
                    unique_classes = set(int(x) for x in np.unique(mask)) - {255}
                    overlap = unique_classes & self.visible_class_ids
                    if len(overlap) > 0:
                        total_valid = (mask != 255).sum()
                        visible_valid = sum((mask == cid).sum() for cid in overlap)
                        if total_valid > 0 and visible_valid / max(total_valid, 1) >= self.min_visible_ratio:
                            indices.append(i)
                except Exception:
                    # Fallback: use standard __getitem__
                    try:
                        sample = self.base_dataset[i]
                        sem_seg = sample["sem_seg"]
                        unique_classes = set(sem_seg.unique().tolist()) - {255}
                        overlap = unique_classes & self.visible_class_ids
                        if len(overlap) > 0:
                            total_valid = (sem_seg != 255).sum().item()
                            visible_valid = sum((sem_seg == cid).sum().item() for cid in overlap)
                            if total_valid > 0 and visible_valid / max(total_valid, 1) >= self.min_visible_ratio:
                                indices.append(i)
                    except Exception:
                        pass
        else:
            # Fallback: standard iteration
            for i in range(n):
                try:
                    sample = self.base_dataset[i]
                    sem_seg = sample["sem_seg"]
                    unique_classes = set(sem_seg.unique().tolist()) - {255}
                    overlap = unique_classes & self.visible_class_ids
                    if len(overlap) > 0:
                        total_valid = (sem_seg != 255).sum().item()
                        visible_valid = sum((sem_seg == cid).sum().item() for cid in overlap)
                        if total_valid > 0 and visible_valid / max(total_valid, 1) >= self.min_visible_ratio:
                            indices.append(i)
                except Exception:
                    pass

        print(f"[ClassFilteredDataset] found {len(indices)}/{n} samples with visible classes", flush=True)
        return indices

    def _ensure_indices(self) -> None:
        if self._indices is None:
            self._indices = self._filter_indices()

    def __len__(self) -> int:
        self._ensure_indices()
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        self._ensure_indices()
        return self.base_dataset[self._indices[idx]]


COCO_STUFF_164_CLASSES = [
    "unlabeled", "aquarium", "banner-stuff", "bedding", "blanket", "bridge", "building-other",
    "bush", "cabinet", "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile",
    "cloth", "clothes", "clouds", "counter-stuff", "cupboard", "curtain-stuff", "cushion",
    "dirt", "door-stuff", "fence-stuff", "floor-marble", "floor-other", "floor-stone",
    "floor-tile", "floor-wood", "flower-stuff", "fog", "food-other", "fruit-stuff", "furniture-other",
    "glass", "gravel", "ground-other", "hill", "house-stuff", "ice", "keyboard-stuff", "light",
    "mat", "moss", "mountain-stuff", "mud", "napkin", "net", "paper", "pavement",
    "pillow", "plant-other", "plastic", "platform", "playingfield", "rug",
    "sand", "sea", "shelf-stuff", "sky-other", "sky-weather", "snow-stuff", "solid-other",
    "spinel", "stairs", "stone-stuff", "straw", "structural-other", "tablecloth",
    "tent", "textile-other", "towel", "tree", "wall-brick", "wall-other",
    "wall-panel", "wall-stone", "wall-tile", "wall-wood", "water-other", "waterdrops",
    "window-blind", "window-other", "wood-stuff",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
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
        augmentation=None,
        num_classes: int = 164,
        visible_class_ids: Optional[List[int]] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.class_names = class_names or COCO_STUFF_164_CLASSES[:num_classes]
        self.transform = transform
        self.augmentation = augmentation
        self.num_classes = num_classes
        self.visible_class_ids = set(visible_class_ids) if visible_class_ids is not None else None

        subdir = self.SPLIT_MAP.get(split, split)
        images_dir = self.root / "images" / subdir
        masks_dir = self.root / "annotations" / subdir

        if not images_dir.exists():
            from covl_seg.data.download import ensure_coco_stuff
            try:
                ensure_coco_stuff(str(self.root))
            except Exception as exc:
                raise FileNotFoundError(
                    f"COCO-Stuff images directory not found: {images_dir}\n"
                    f"Auto-download failed: {exc}\n"
                    f"Run: python -m covl_seg.data.download --dataset coco-stuff"
                ) from exc
            if not images_dir.exists():
                raise FileNotFoundError(
                    f"COCO-Stuff images directory not found: {images_dir}\n"
                    f"Run: python -m covl_seg.data.download --dataset coco-stuff"
                )

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
        sem_seg[sem_seg >= self.num_classes] = 255

        if self.visible_class_ids is not None:
            valid_mask = torch.zeros_like(sem_seg, dtype=torch.bool)
            for cid in self.visible_class_ids:
                valid_mask |= (sem_seg == cid)
            sem_seg[~valid_mask & (sem_seg != 255)] = 255

        if self.augmentation is not None:
            image_tensor, sem_seg = self.augmentation(image_tensor, sem_seg)
        elif self.transform is not None:
            image_tensor = self.transform(image_tensor)

        return {
            "image": image_tensor,
            "sem_seg": sem_seg,
            "class_names": self.class_names,
            "image_id": info["image_id"],
        }