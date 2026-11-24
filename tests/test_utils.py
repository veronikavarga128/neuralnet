"""Tests for unet_pet.utils."""

# The legacy global NumPy RNG is what set_seed / seed_worker seed, so the tests use it.
# ruff: noqa: NPY002

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split

from unet_pet.utils import (
    get_device,
    make_generator,
    project_root,
    resolve_path,
    seed_worker,
    set_seed,
)


def _draw() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.rand()), torch.rand(3)


def test_set_seed_makes_all_generators_repeatable() -> None:
    set_seed(42)
    a = _draw()
    set_seed(42)
    b = _draw()
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert torch.equal(a[2], b[2])


def test_different_seeds_give_different_numbers() -> None:
    set_seed(1)
    a = torch.rand(3)
    set_seed(2)
    assert not torch.equal(a, torch.rand(3))


def test_set_seed_configures_cudnn() -> None:
    set_seed(0)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_seeded_split_is_repeatable() -> None:
    items = list(range(100))
    first = random_split(items, [85, 15], generator=make_generator(42))
    second = random_split(items, [85, 15], generator=make_generator(42))
    assert first[1].indices == second[1].indices
    third = random_split(items, [85, 15], generator=make_generator(43))
    assert first[1].indices != third[1].indices


def test_seed_worker_derives_numpy_seed_from_torch() -> None:
    torch.manual_seed(123)
    seed_worker(0)
    a = (np.random.rand(), random.random())
    torch.manual_seed(123)
    seed_worker(0)
    assert a == (np.random.rand(), random.random())


def test_get_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert get_device() == torch.device("cpu")


def test_get_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_device().type == "cuda"


def test_get_device_explicit_preference() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_project_root_and_resolve_path(tmp_path: Path) -> None:
    root = project_root()
    assert (root / "unet_pet" / "__init__.py").is_file()
    assert resolve_path("configs/default.yaml") == root / "configs" / "default.yaml"
    assert resolve_path(tmp_path) == tmp_path
