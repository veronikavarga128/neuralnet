"""Tests for unet_pet.data — no download: a tiny fake base dataset stands in for Oxford-IIIT Pet."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from unet_pet.config import load_config
from unet_pet.data import (
    JointTransform,
    PetSegmentationDataset,
    build_transform,
    class_pixel_fractions,
    denormalize,
    make_dataloaders,
    split_indices,
    trimap_to_labels,
)

RED, BLUE = (255, 0, 0), (0, 0, 255)


class FakePetBase(Dataset):
    """Returns (PIL RGB image, PIL trimap mask) like OxfordIIITPet(target_types='segmentation').

    Left half: red image, mask value 1 (pet). Right half: blue image, mask value 3 (border).
    Odd sizes and non-square shapes exercise the resize.
    """

    def __init__(self, n: int = 20, h: int = 50, w: int = 70) -> None:
        self.n, self.h, self.w = n, h, w

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        mask = np.full((self.h, self.w), 3, dtype=np.uint8)
        half = self.w // 2
        img[:, :half] = RED
        img[:, half:] = BLUE
        mask[:, :half] = 1
        return Image.fromarray(img), Image.fromarray(mask)


@pytest.fixture
def cfg():
    return load_config(overrides={"data.img_size": 32, "train.batch_size": 4})


def test_output_shapes_dtypes_and_labels(cfg) -> None:
    ds = PetSegmentationDataset(FakePetBase(), build_transform(cfg, train=True))
    img, mask = ds[0]
    assert img.shape == (3, 32, 32) and img.dtype == torch.float32
    assert mask.shape == (32, 32) and mask.dtype == torch.int64
    assert set(mask.unique().tolist()) <= {0, 1, 2}
    assert type(img) is torch.Tensor and type(mask) is torch.Tensor  # no tv_tensor subclasses


def test_resize_uses_nearest_for_mask() -> None:
    """A 1/3 checkerboard must never produce the in-between value 2 (bilinear would)."""
    tf = JointTransform(img_size=48, mean=[0, 0, 0], std=[1, 1, 1], train=False)
    checker = ((np.indices((37, 53)).sum(axis=0) % 2) * 2 + 1).astype(np.uint8)  # values 1, 3
    _, raw = tf(Image.new("RGB", (53, 37)), Image.fromarray(checker))
    assert set(raw.unique().tolist()) <= {1, 3}


def test_flip_is_applied_to_image_and_mask_together(cfg) -> None:
    flip_cfg = load_config(
        overrides={"data.img_size": 32, "data.hflip_prob": 1.0, "data.color_jitter": [0, 0, 0, 0]}
    )
    img, mask = PetSegmentationDataset(FakePetBase(), build_transform(flip_cfg, train=True))[0]
    rgb = denormalize(img, flip_cfg.data.mean, flip_cfg.data.std)
    # After the flip the right half is red (pet, label 0) and the left half blue (border, 2).
    assert (mask[:, -4:] == 0).all() and (mask[:, :4] == 2).all()
    assert rgb[0, :, -4:].mean() > 0.9 and rgb[2, :, -4:].mean() < 0.1
    assert rgb[2, :, :4].mean() > 0.9 and rgb[0, :, :4].mean() < 0.1


def test_eval_transform_does_not_augment(cfg) -> None:
    eval_cfg = load_config(overrides={"data.img_size": 32, "data.hflip_prob": 1.0})
    tf = build_transform(eval_cfg, train=False)
    img, mask = PetSegmentationDataset(FakePetBase(), tf)[0]
    assert (mask[:, :4] == 0).all() and (mask[:, -4:] == 2).all()  # not flipped
    again, _ = PetSegmentationDataset(FakePetBase(), tf)[0]
    assert torch.equal(img, again)  # no random colour jitter either


def test_color_jitter_touches_image_only() -> None:
    jitter_cfg = load_config(
        overrides={
            "data.img_size": 32,
            "data.hflip_prob": 0.0,
            "data.color_jitter": [0.9, 0.9, 0.9, 0.3],
        }
    )
    train_ds = PetSegmentationDataset(FakePetBase(), build_transform(jitter_cfg, train=True))
    eval_ds = PetSegmentationDataset(FakePetBase(), build_transform(jitter_cfg, train=False))
    torch.manual_seed(0)
    img_t, mask_t = train_ds[0]
    img_e, mask_e = eval_ds[0]
    assert torch.equal(mask_t, mask_e)
    assert not torch.allclose(img_t, img_e)


def test_trimap_to_labels_maps_and_validates() -> None:
    out = trimap_to_labels(torch.tensor([[1, 2, 3]], dtype=torch.uint8))
    assert out.tolist() == [[0, 1, 2]] and out.dtype == torch.int64
    with pytest.raises(ValueError):
        trimap_to_labels(torch.tensor([0, 1], dtype=torch.uint8))  # 0 would wrap on uint8
    with pytest.raises(ValueError):
        trimap_to_labels(torch.tensor([4]))


def test_split_is_deterministic_disjoint_and_complete() -> None:
    train_a, val_a = split_indices(3680, 0.15, seed=42)
    train_b, val_b = split_indices(3680, 0.15, seed=42)
    assert (train_a, val_a) == (train_b, val_b)
    assert not set(train_a) & set(val_a)
    assert sorted(train_a + val_a) == list(range(3680))
    assert len(val_a) == 552
    assert split_indices(3680, 0.15, seed=43)[1] != val_a


def test_make_dataloaders_with_fake_base(cfg) -> None:
    loaders = make_dataloaders(cfg, trainval_base=FakePetBase(20), test_base=FakePetBase(6))
    assert len(loaders["train"].dataset) == 17 and len(loaders["val"].dataset) == 3
    assert len(loaders["test"].dataset) == 6
    # train shuffling is seeded: the first epoch of two freshly built loaders matches
    again = make_dataloaders(cfg, trainval_base=FakePetBase(20), test_base=FakePetBase(6))
    epoch1 = list(loaders["train"].sampler)
    assert epoch1 == list(again["train"].sampler)
    # ...while consecutive epochs of one loader are shuffled differently
    assert epoch1 != list(loaders["train"].sampler)

    images, masks = next(iter(loaders["train"]))
    assert images.shape == (4, 3, 32, 32) and masks.shape == (4, 32, 32)
    assert masks.dtype == torch.int64
    assert len(loaders["train"]) == 4  # 17 // 4 with drop_last


def test_denormalize_inverts_normalize() -> None:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    x = torch.rand(2, 3, 8, 8)
    normed = (x - torch.tensor(mean).view(-1, 1, 1)) / torch.tensor(std).view(-1, 1, 1)
    assert torch.allclose(denormalize(normed, mean, std), x, atol=1e-6)


def test_class_pixel_fractions(cfg) -> None:
    ds = PetSegmentationDataset(FakePetBase(4), build_transform(cfg, train=False))
    fractions = class_pixel_fractions(ds)
    assert fractions.sum().item() == pytest.approx(1.0)
    assert fractions[1].item() == 0.0  # the fake data has no background
    assert fractions[0].item() == pytest.approx(0.5, abs=0.05)
