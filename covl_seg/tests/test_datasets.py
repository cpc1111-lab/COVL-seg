import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from covl_seg.data.datasets import ADE20KDataset, COCOStuffDataset


def _make_mock_ade20k(tmp_path, split, num_images=3, size=(64, 64)):
    images_dir = tmp_path / "images" / split
    masks_dir = tmp_path / "annotations" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
        img.save(images_dir / f"scene_{i:04d}.jpg")
        mask = Image.fromarray(np.random.randint(0, 5, size, dtype=np.uint8))
        mask.save(masks_dir / f"scene_{i:04d}.png")


class TestADE20KDataset:
    def test_dataset_length(self, tmp_path):
        _make_mock_ade20k(tmp_path, "training", num_images=3)
        ds = ADE20KDataset(root=str(tmp_path), split="training")
        assert len(ds) == 3

    def test_excludes_missing_mask(self, tmp_path):
        images_dir = tmp_path / "images" / "training"
        masks_dir = tmp_path / "annotations" / "training"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        img = Image.new("RGB", (64, 64))
        img.save(images_dir / "has_mask.jpg")
        mask = Image.new("L", (64, 64), color=1)
        mask.save(masks_dir / "has_mask.png")
        img.save(images_dir / "no_mask.jpg")

        ds = ADE20KDataset(root=str(tmp_path), split="training")
        assert len(ds) == 1

    def test_return_types_and_shapes(self, tmp_path):
        _make_mock_ade20k(tmp_path, "training", num_images=2, size=(48, 64))
        ds = ADE20KDataset(
            root=str(tmp_path),
            split="training",
            class_names=["wall", "floor", "ceiling"],
        )
        sample = ds[0]
        assert isinstance(sample, dict)
        assert set(sample.keys()) == {"image", "sem_seg", "class_names", "image_id"}

        image = sample["image"]
        assert isinstance(image, torch.Tensor)
        assert image.dtype == torch.float32
        assert image.shape == (3, 48, 64)
        assert image.min() >= 0.0
        assert image.max() <= 1.0

        sem_seg = sample["sem_seg"]
        assert isinstance(sem_seg, torch.Tensor)
        assert sem_seg.dtype == torch.int64
        assert sem_seg.shape == (48, 64)

        assert sample["class_names"] == ["wall", "floor", "ceiling"]
        assert isinstance(sample["image_id"], str)

    def test_different_split(self, tmp_path):
        _make_mock_ade20k(tmp_path, "validation", num_images=2)
        ds = ADE20KDataset(
            root=str(tmp_path),
            split="validation",
            class_names=["sky"],
        )
        assert len(ds) == 2
        sample = ds[0]
        assert sample["class_names"] == ["sky"]

    def test_class_names_default_none(self, tmp_path):
        _make_mock_ade20k(tmp_path, "training", num_images=1)
        ds = ADE20KDataset(root=str(tmp_path), split="training")
        sample = ds[0]
        assert sample["class_names"] is None

    def test_image_id_is_stem(self, tmp_path):
        _make_mock_ade20k(tmp_path, "training", num_images=1)
        ds = ADE20KDataset(root=str(tmp_path), split="training")
        sample = ds[0]
        assert sample["image_id"] == "scene_0000"


class TestADE20KDatasetNormalization:
    def test_pixels_divided_by_255(self, tmp_path):
        images_dir = tmp_path / "images" / "training"
        masks_dir = tmp_path / "annotations" / "training"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        img_arr = np.full((32, 48, 3), 255, dtype=np.uint8)
        Image.fromarray(img_arr).save(images_dir / "white.jpg")
        mask_arr = np.zeros((32, 48), dtype=np.uint8)
        Image.fromarray(mask_arr).save(masks_dir / "white.png")

        ds = ADE20KDataset(root=str(tmp_path), split="training")
        sample = ds[0]
        assert torch.allclose(sample["image"], torch.ones(3, 32, 48))


class TestCOCOStuffDataset:
    def test_coco_stuff_loads_images_and_masks(self, tmp_path):
        img_dir = tmp_path / "images" / "train2017"
        ann_dir = tmp_path / "annotations" / "train2017"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        img.save(img_dir / "000000000009.jpg")
        mask = Image.new("L", (64, 64), color=1)
        mask.save(ann_dir / "000000000009.png")

        ds = COCOStuffDataset(root=str(tmp_path), split="training", num_classes=164)
        assert len(ds) == 1
        sample = ds[0]
        assert "image" in sample
        assert "sem_seg" in sample
        assert sample["image"].shape == (3, 64, 64)

    def test_coco_stuff_validation_split(self, tmp_path):
        img_dir = tmp_path / "images" / "val2017"
        ann_dir = tmp_path / "annotations" / "val2017"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        img = Image.new("RGB", (32, 32))
        img.save(img_dir / "000000000139.jpg")
        mask = Image.new("L", (32, 32), color=0)
        mask.save(ann_dir / "000000000139.png")

        ds = COCOStuffDataset(root=str(tmp_path), split="validation", num_classes=164)
        assert len(ds) == 1
        sample = ds[0]
        assert sample["image"].shape == (3, 32, 32)

    def test_coco_stuff_clamps_labels(self, tmp_path):
        img_dir = tmp_path / "images" / "train2017"
        ann_dir = tmp_path / "annotations" / "train2017"
        img_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)

        img = Image.new("RGB", (32, 32))
        img.save(img_dir / "000000000009.jpg")
        mask = Image.new("L", (32, 32), color=200)
        mask.save(ann_dir / "000000000009.png")

        ds = COCOStuffDataset(root=str(tmp_path), split="training", num_classes=164)
        sample = ds[0]
        out_of_range = (sample["sem_seg"] >= 164) & (sample["sem_seg"] != 255)
        assert out_of_range.sum() == 0, "Out-of-range labels should be mapped to 255"
        assert (sample["sem_seg"] == 255).all(), "All pixels with value 200 (>=164) should become 255"