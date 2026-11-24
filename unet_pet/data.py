"""Oxford-IIIT Pet dataset wrapper, paired image/mask transforms, split and DataLoaders.

Label convention used everywhere in the project (original trimap value minus one):
``0 = pet, 1 = background, 2 = border``.
"""

from __future__ import annotations

import argparse
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from unet_pet.config import DEFAULT_CONFIG_PATH, Config, load_config
from unet_pet.utils import make_generator, resolve_path, seed_worker

CLASS_NAMES: tuple[str, ...] = ("pet", "background", "border")
SAMPLE_DIR = "sample_data"


class JointTransform:
    """Apply the same geometric transform to an image and its mask.

    The mask is wrapped in ``tv_tensors.Mask``. ``transforms.v2`` then (1) resizes it with
    NEAREST interpolation, so no in-between, non-existent labels appear, (2) flips it with
    the very same random decision as the image, and (3) leaves it untouched in the
    photometric steps (ColorJitter, ToDtype, Normalize), which only act on images.
    """

    def __init__(
        self,
        img_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        train: bool,
        hflip_prob: float = 0.0,
        color_jitter: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
    ) -> None:
        steps: list[Any] = [v2.Resize((img_size, img_size), antialias=True)]
        if train:
            if hflip_prob > 0:
                steps.append(v2.RandomHorizontalFlip(p=hflip_prob))
            if any(v > 0 for v in color_jitter):
                brightness, contrast, saturation, hue = color_jitter
                steps.append(
                    v2.ColorJitter(
                        brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
                    )
                )
        steps += [
            v2.ToDtype(torch.float32, scale=True),  # image only: uint8 [0,255] -> float [0,1]
            v2.Normalize(mean=list(mean), std=list(std)),  # image only
        ]
        self.pipeline = v2.Compose(steps)

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(image float32 (3,H,W), raw trimap mask uint8 (H,W))``."""
        img = F.to_image(image.convert("RGB"))
        msk = tv_tensors.Mask(torch.from_numpy(np.array(mask, dtype=np.uint8)))
        img, msk = self.pipeline(img, msk)
        return img.as_subclass(torch.Tensor), msk.as_subclass(torch.Tensor)


def build_transform(cfg: Config, train: bool) -> JointTransform:
    """Create the train (augmenting) or eval (resize + normalise only) transform from config."""
    d = cfg.data
    return JointTransform(
        img_size=d.img_size,
        mean=d.mean,
        std=d.std,
        train=train,
        hflip_prob=d.hflip_prob,
        color_jitter=d.color_jitter,
    )


def trimap_to_labels(mask: torch.Tensor) -> torch.Tensor:
    """Map trimap values ``{1,2,3}`` to class labels ``{0,1,2}`` as ``int64``.

    Converts to ``long`` *before* subtracting: on ``uint8`` a stray 0 would wrap to 255.
    """
    labels = mask.long() - 1
    if labels.numel() and (labels.min() < 0 or labels.max() >= len(CLASS_NAMES)):
        found = torch.unique(mask).tolist()
        raise ValueError(f"Unexpected trimap values {found}; expected a subset of {{1, 2, 3}}")
    return labels


class PetSegmentationDataset(Dataset):
    """``(image, mask)`` pairs from a base dataset, restricted to ``indices``.

    ``base`` must return ``(PIL image, PIL trimap mask)``, as ``OxfordIIITPet`` with
    ``target_types="segmentation"`` and no transforms does. Two wrappers over the same base
    with different transforms give an augmented train set and a clean validation set.
    """

    def __init__(
        self,
        base: Dataset,
        transform: JointTransform,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.base = base
        self.transform = transform
        self.indices = list(range(len(base))) if indices is None else list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.base[self.indices[idx]]
        img, raw_mask = self.transform(image, mask)
        return img, trimap_to_labels(raw_mask)


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Seeded, disjoint train/val split of ``range(n)``. Returns sorted index lists."""
    if n < 2:
        raise ValueError("Need at least 2 samples to split")
    n_val = min(max(1, round(n * val_fraction)), n - 1)
    perm = torch.randperm(n, generator=make_generator(seed)).tolist()
    return sorted(perm[n_val:]), sorted(perm[:n_val])


def load_base_datasets(cfg: Config, download: bool = True) -> tuple[Dataset, Dataset]:
    """Return the raw official ``trainval`` and ``test`` splits (downloaded on first use)."""
    root = cfg.data.root
    kwargs: dict[str, Any] = {"target_types": "segmentation", "download": download}
    return (
        OxfordIIITPet(root=root, split="trainval", **kwargs),
        OxfordIIITPet(root=root, split="test", **kwargs),
    )


def make_datasets(
    cfg: Config, trainval_base: Dataset, test_base: Dataset
) -> dict[str, PetSegmentationDataset]:
    """Build ``train`` / ``val`` / ``test`` datasets; only ``train`` is augmented."""
    train_idx, val_idx = split_indices(len(trainval_base), cfg.data.val_fraction, cfg.seed)
    eval_tf = build_transform(cfg, train=False)
    return {
        "train": PetSegmentationDataset(trainval_base, build_transform(cfg, train=True), train_idx),
        "val": PetSegmentationDataset(trainval_base, eval_tf, val_idx),
        "test": PetSegmentationDataset(test_base, eval_tf),
    }


def make_dataloaders(
    cfg: Config,
    trainval_base: Dataset | None = None,
    test_base: Dataset | None = None,
    download: bool = True,
) -> dict[str, DataLoader]:
    """Create seeded ``train`` / ``val`` / ``test`` DataLoaders.

    The base datasets can be injected (tests use a small fake one); otherwise the
    Oxford-IIIT Pet splits are loaded from ``cfg.data.root`` and downloaded if needed.
    """
    if trainval_base is None or test_base is None:
        trainval_base, test_base = load_base_datasets(cfg, download=download)
    datasets = make_datasets(cfg, trainval_base, test_base)

    workers = cfg.data.num_workers
    common: dict[str, Any] = {
        "batch_size": cfg.train.batch_size,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker,
    }
    return {
        # drop_last: a final batch of size 1 would break BatchNorm in train mode.
        "train": DataLoader(
            datasets["train"],
            shuffle=True,
            drop_last=len(datasets["train"]) > cfg.train.batch_size,
            generator=make_generator(cfg.seed),
            **common,
        ),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def denormalize(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Undo normalisation for display: ``(..., 3, H, W)`` -> values clipped to ``[0, 1]``."""
    m = torch.as_tensor(mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    s = torch.as_tensor(std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
    return (image * s + m).clamp(0.0, 1.0)


def class_pixel_fractions(
    dataset: Dataset, num_classes: int = len(CLASS_NAMES), max_items: int | None = None
) -> torch.Tensor:
    """Fraction of pixels per class over (the first ``max_items`` of) a dataset."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    n = len(dataset) if max_items is None else min(max_items, len(dataset))
    for i in range(n):
        _, mask = dataset[i]
        counts += torch.bincount(mask.flatten(), minlength=num_classes)[:num_classes]
    return counts.double() / counts.sum().clamp(min=1)


# --------------------------------------------------------------------------------------
# sample_data/: a few images + masks committed to git for the offline notebook demo
# --------------------------------------------------------------------------------------


class SampleFolder(Dataset):
    """``(PIL image, PIL trimap mask)`` pairs from ``<root>/images/*.jpg`` + ``<root>/masks/*.png``.

    Mirrors the raw ``OxfordIIITPet`` output, so it plugs into ``PetSegmentationDataset``
    and goes through exactly the same transforms as the real data.
    """

    def __init__(self, root: str | Path = SAMPLE_DIR) -> None:
        self.root = resolve_path(root)
        self.image_paths = sorted((self.root / "images").glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.root / 'images'}")
        self.mask_paths = [self.root / "masks" / f"{p.stem}.png" for p in self.image_paths]
        missing = [p.name for p in self.mask_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing masks in {self.root / 'masks'}: {missing}")
        self.names = [p.stem for p in self.image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        with Image.open(self.image_paths[idx]) as img, Image.open(self.mask_paths[idx]) as mask:
            return img.convert("RGB"), mask.copy()


def load_sample_data(
    cfg: Config, root: str | Path = SAMPLE_DIR
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load ``sample_data/`` as model-ready tensors (eval transform, no augmentation).

    Returns:
        images ``float32 (N, 3, S, S)`` normalised, masks ``int64 (N, S, S)`` with labels
        ``{0,1,2}``, and the file stems (e.g. ``"Abyssinian_201"``; the breed is the prefix).
    """
    folder = SampleFolder(root)
    ds = PetSegmentationDataset(folder, build_transform(cfg, train=False))
    images, masks = zip(*(ds[i] for i in range(len(ds))), strict=True)
    return torch.stack(images), torch.stack(masks), folder.names


def _breed(image_path: Path) -> str:
    """``Abyssinian_201.jpg`` -> ``Abyssinian`` (file names encode the breed)."""
    return re.sub(r"_\d+$", "", image_path.stem)


def export_sample_data(
    data_root: str | Path,
    out_dir: str | Path = SAMPLE_DIR,
    per_species: int = 4,
    max_side: int = 256,
    seed: int = 42,
    jpeg_quality: int = 85,
) -> list[str]:
    """Copy a small, reproducible demo subset of the official *test* split to ``out_dir``.

    Test images are used because the model never sees them during training, so the
    offline demo shows honest predictions. ``per_species`` cats and dogs are drawn with a
    fixed seed, each from a different breed. Images are downscaled (longer side
    ``max_side``) to keep the folder small; masks keep the raw trimap values ``{1,2,3}`` and
    are resized with NEAREST so no fake labels appear.
    """
    base = OxfordIIITPet(root=data_root, split="test", target_types="binary-category")
    rng = random.Random(seed)
    order = list(range(len(base)))
    rng.shuffle(order)

    chosen: list[int] = []
    breeds: set[str] = set()
    for species in range(len(base.bin_classes)):  # 0 = Cat, 1 = Dog
        picked = 0
        for i in order:
            breed = _breed(base._images[i])
            if base._bin_labels[i] == species and breed not in breeds:
                chosen.append(i)
                breeds.add(breed)
                picked += 1
                if picked == per_species:
                    break

    out = resolve_path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    names = []
    for i in chosen:
        img_path, mask_path = Path(base._images[i]), Path(base._segs[i])
        with Image.open(img_path) as img, Image.open(mask_path) as mask:
            img = img.convert("RGB")
            scale = min(1.0, max_side / max(img.size))
            size = (round(img.width * scale), round(img.height * scale))
            img.resize(size, Image.Resampling.BICUBIC).save(
                out / "images" / f"{img_path.stem}.jpg", quality=jpeg_quality, optimize=True
            )
            mask.resize(size, Image.Resampling.NEAREST).save(
                out / "masks" / f"{img_path.stem}.png", optimize=True
            )
        names.append(img_path.stem)
    return names


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ``python -m unet_pet.data --export-sample`` regenerates ``sample_data/``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--export-sample", action="store_true", help="(re)create sample_data/")
    parser.add_argument("--out", default=SAMPLE_DIR)
    parser.add_argument("--per-species", type=int, default=4)
    args = parser.parse_args(argv)
    if not args.export_sample:
        parser.print_help()
        return
    cfg = load_config(args.config)
    names = export_sample_data(cfg.data.root, args.out, args.per_species, seed=cfg.seed)
    out = resolve_path(args.out)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"Wrote {len(names)} samples to {out} ({total / 1024:.0f} KB): {', '.join(names)}")


if __name__ == "__main__":
    main()
