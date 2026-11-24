"""Configuration loading: YAML file -> typed ``Config`` dataclass.

The YAML file is the single source of truth for hyperparameters: the dataclasses below
have no default values, so a missing or misspelled key fails loudly at load time instead
of silently falling back to a value hidden in the code.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from unet_pet.utils import resolve_path

DEFAULT_CONFIG_PATH = "configs/default.yaml"
SCHEDULERS = ("reduce_on_plateau", "none")


@dataclass
class DataConfig:
    root: Path
    img_size: int
    val_fraction: float
    num_workers: int
    mean: list[float]
    std: list[float]


@dataclass
class ModelConfig:
    in_channels: int
    num_classes: int
    base_channels: int
    depth: int


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    scheduler: str
    dice_weight: float
    log_images_every: int


@dataclass
class PathsConfig:
    runs: Path
    checkpoints: Path
    best_weights: Path


@dataclass
class Config:
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    paths: PathsConfig
    source: Path | None = field(default=None, compare=False)  # file the config came from

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` if any value is outside its sensible range."""
        d, m, t = self.data, self.model, self.train
        checks = [
            (0.0 < d.val_fraction < 1.0, "data.val_fraction must be in (0, 1)"),
            (d.img_size > 0, "data.img_size must be positive"),
            (d.img_size % 2**m.depth == 0, "data.img_size must be divisible by 2**model.depth"),
            (d.num_workers >= 0, "data.num_workers must be >= 0"),
            (len(d.mean) == len(d.std) == m.in_channels, "data.mean/std need in_channels values"),
            (all(s > 0 for s in d.std), "data.std values must be positive"),
            (m.num_classes >= 2, "model.num_classes must be >= 2"),
            (m.base_channels > 0 and m.depth > 0, "model.base_channels and depth must be > 0"),
            (t.epochs > 0 and t.batch_size > 0, "train.epochs and batch_size must be > 0"),
            (t.lr > 0 and t.weight_decay >= 0, "train.lr must be > 0, weight_decay >= 0"),
            (t.dice_weight >= 0, "train.dice_weight must be >= 0"),
            (t.scheduler in SCHEDULERS, f"train.scheduler must be one of {SCHEDULERS}"),
            (t.log_images_every >= 0, "train.log_images_every must be >= 0"),
        ]
        errors = [msg for ok, msg in checks if not ok]
        if errors:
            raise ValueError("Invalid configuration: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        """Plain, YAML/JSON-serialisable dict (paths become strings)."""

        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return value.as_posix()
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value

        raw = dataclasses.asdict(self)
        raw.pop("source")
        return convert(raw)


_SECTIONS: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "paths": PathsConfig,
}


def _coerce(value: Any, target: type | str, key: str) -> Any:
    """Convert ``value`` (possibly a CLI string) to the annotated field type."""
    target = str(target)
    try:
        if target == "bool":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered not in {"true", "false", "1", "0", "yes", "no"}:
                    raise ValueError(value)
                return lowered in {"true", "1", "yes"}
            return bool(value)
        if target == "int":
            if isinstance(value, float) and not value.is_integer():
                raise ValueError(value)
            return int(value)
        if target == "float":
            return float(value)
        if target == "str":
            return str(value)
        if target == "Path":
            return resolve_path(value)
        if target == "list[float]":
            if isinstance(value, str):
                value = yaml.safe_load(value)
            return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Config key '{key}': cannot convert {value!r} to {target}") from exc
    raise TypeError(f"Config key '{key}': unsupported field type {target}")


def _build_section(cls: type, raw: Any, section: str) -> Any:
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config section '{section}' must be a mapping")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    missing = set(fields) - set(raw)
    if unknown:
        raise KeyError(f"Unknown key(s) in '{section}': {sorted(unknown)}")
    if missing:
        raise KeyError(f"Missing key(s) in '{section}': {sorted(missing)}")
    values = {name: _coerce(raw[name], f.type, f"{section}.{name}") for name, f in fields.items()}
    return cls(**values)


def _apply_overrides(raw: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Apply dotted-key overrides (e.g. ``{"train.epochs": 1}``) to the raw YAML dict."""
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node = raw
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                raise KeyError(f"Unknown config key: '{dotted}'")
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key: '{dotted}'")
        node[parts[-1]] = value


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Load a YAML config into a validated ``Config``.

    Args:
        path: YAML file; a relative path is resolved against the repository root.
        overrides: optional dotted-key overrides, e.g. ``{"train.epochs": "1"}``. String
            values (as they come from the command line) are converted to the field type.

    Raises:
        FileNotFoundError: the file does not exist.
        KeyError: a key is missing, unknown, or an override targets a non-existent key.
        ValueError: a value cannot be converted or is out of range.
    """
    config_path = resolve_path(path)
    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping at the top level")

    if overrides:
        _apply_overrides(raw, overrides)

    expected = {"seed", *_SECTIONS}
    unknown, missing = set(raw) - expected, expected - set(raw)
    if unknown:
        raise KeyError(f"Unknown top-level config key(s): {sorted(unknown)}")
    if missing:
        raise KeyError(f"Missing top-level config key(s): {sorted(missing)}")

    sections = {name: _build_section(cls, raw[name], name) for name, cls in _SECTIONS.items()}
    return Config(seed=_coerce(raw["seed"], "int", "seed"), source=config_path, **sections)
