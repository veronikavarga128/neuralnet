"""Reproducibility and environment helpers: seeding, device selection, paths."""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch


def project_root() -> Path:
    """Return the repository root (the directory that contains the ``unet_pet`` package)."""
    return Path(__file__).resolve().parent.parent


def resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` against the repository root unless it is already absolute.

    Paths are anchored to the repo root rather than the current working directory,
    so the notebook, the CLI and the tests all see the same files.
    """
    p = Path(path).expanduser()
    return p if p.is_absolute() else project_root() / p


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every random number generator used in the project.

    Covers Python ``random``, NumPy and PyTorch (CPU and all CUDA devices). With
    ``deterministic=True`` cuDNN is forced to deterministic kernels and its autotuner is
    disabled. We deliberately do not call ``torch.use_deterministic_algorithms(True)``:
    some CUDA ops used by a U-Net (e.g. the 2D cross-entropy backward) have no
    deterministic implementation and would raise, while the cuDNN flags already give
    run-to-run reproducibility that is good enough for this project.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - the global RNG is exactly what we must seed
    torch.manual_seed(seed)  # also seeds every CUDA device
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for ``DataLoader``: seed NumPy and ``random`` in each worker.

    PyTorch already gives every worker a distinct torch seed derived from the loader's
    generator; this propagates it to NumPy and ``random`` so augmentations that use those
    libraries are reproducible too. ``worker_id`` is unused but required by the API.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002 - seeds the worker's global RNG
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a ``torch.Generator`` seeded with ``seed`` (for DataLoader shuffling and splits)."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_device(preferred: str | None = None) -> torch.device:
    """Pick the best available device: CUDA, then Apple MPS, then CPU.

    Args:
        preferred: optional explicit device string (e.g. ``"cpu"``) that bypasses the
            automatic choice.
    """
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_overrides(pairs: Sequence[str]) -> dict[str, str]:
    """``["train.lr=0.01", ...]`` (CLI ``--set`` values) -> ``{"train.lr": "0.01", ...}``."""
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"--set expects key=value, got '{pair}'")
        overrides[key.strip()] = value.strip()
    return overrides


def positive_int(text: str) -> int:
    """``argparse`` type for options that must be >= 1."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value
