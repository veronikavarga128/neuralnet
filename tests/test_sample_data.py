"""Tests for the committed sample_data/ folder and its loader (offline, no download)."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from unet_pet.config import load_config
from unet_pet.data import SampleFolder, load_sample_data
from unet_pet.utils import project_root

SAMPLE_ROOT = project_root() / "sample_data"


@pytest.fixture
def cfg():
    return load_config()


def test_sample_folder_is_small_and_complete() -> None:
    files = [p for p in SAMPLE_ROOT.rglob("*") if p.is_file()]
    assert sum(p.stat().st_size for p in files) < 2 * 1024 * 1024
    images = sorted(p.stem for p in (SAMPLE_ROOT / "images").glob("*.jpg"))
    masks = sorted(p.stem for p in (SAMPLE_ROOT / "masks").glob("*.png"))
    assert 6 <= len(images) <= 8
    assert images == masks


def test_sample_masks_are_raw_trimaps_matching_image_size() -> None:
    folder = SampleFolder()
    for i in range(len(folder)):
        image, mask = folder[i]
        assert image.size == mask.size
        assert set(np.unique(np.array(mask)).tolist()) <= {1, 2, 3}


def test_sample_has_cats_and_dogs() -> None:
    # Oxford-IIIT Pet naming: cat breeds are Capitalised, dog breeds lower-case.
    first_letters = {name[0] for name in SampleFolder().names}
    assert any(c.isupper() for c in first_letters) and any(c.islower() for c in first_letters)


def test_load_sample_data_returns_model_ready_tensors(cfg) -> None:
    images, masks, names = load_sample_data(cfg)
    n, s = len(names), cfg.data.img_size
    assert images.shape == (n, 3, s, s) and images.dtype == torch.float32
    assert masks.shape == (n, s, s) and masks.dtype == torch.int64
    assert set(masks.unique().tolist()) == {0, 1, 2}


def test_load_sample_data_is_deterministic(cfg) -> None:
    a = load_sample_data(cfg)
    b = load_sample_data(cfg)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]


def _make_folder(root: Path, stems: list[str], with_masks: bool = True) -> None:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for stem in stems:
        Image.new("RGB", (40, 30), (10, 20, 30)).save(root / "images" / f"{stem}.jpg")
        if with_masks:
            Image.fromarray(np.full((30, 40), 2, np.uint8)).save(root / "masks" / f"{stem}.png")


def test_custom_folder_loads(tmp_path: Path, cfg) -> None:
    _make_folder(tmp_path, ["a_1", "b_2"])
    images, masks, names = load_sample_data(cfg, tmp_path)
    assert names == ["a_1", "b_2"] and images.shape[0] == 2
    assert (masks == 1).all()  # trimap 2 (background) -> label 1


def test_missing_mask_raises(tmp_path: Path) -> None:
    _make_folder(tmp_path, ["a_1"], with_masks=False)
    with pytest.raises(FileNotFoundError, match="a_1.png"):
        SampleFolder(tmp_path)


def test_empty_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SampleFolder(tmp_path)
