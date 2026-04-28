import numpy as np
import pytest
import torch
from PIL import Image

from covl_seg.data.datasets import ADE20KDataset


def _make_mock_ade20k(root, split, num_images=3, size=(64, 48)):
    images_dir = root / "images" / split
    masks_dir = root / "annotations" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        img = Image.fromarray(
            np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        )
        img.save(images_dir / f"scene_{i:04d}.jpg")

        mask = Image.fromarray(
            np.random.randint(0, 5, size, dtype=np.uint8), mode="L"
        )
        mask.save(masks_dir / f"scene_{i:04d}.png")


class TestADE20KDatasetInit:
    def test_length_matches_valid_pairs(self, tmp_path):
        _make_mock_ade20k(tmp_path, "training", num_images=4)
        ds = ADE20KDataset(
            root=str(tmp_path), split="training", class_names=["a", "b", "c"]
        )
        assert len(ds) == 4

    def test_missing_mask_excluded(self, tmp_path):
        _make_mock_ade20k(tmp_path, "training", num_images=3)
        masks_dir = tmp_path / "annotations" / "training"
        (masks_dir / "scene_0001.png").unlink()
        ds = ADE20KDataset(
            root=str(tmp_path), split="training"
        )
        assert len(ds) == 2

    def test_missing_image_excluded(self, tmp_path):
        _make_mock_ade20k(tmp_path, "training", num_images=3)
        images_dir = tmp_path / "images" / "training"
        (images_dir / "scene_0001.jpg").unlink()
        ds = ADE20KDataset(
            root=str(tmp_path), split="training"
        )
        assert len(ds) == 2


class TestADE20KDatasetGetitem:
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