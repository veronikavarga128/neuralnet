"""Tests for unet_pet.config."""

from pathlib import Path

import pytest
import yaml

from unet_pet.config import Config, load_config
from unet_pet.utils import project_root


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def raw_default() -> dict:
    with (project_root() / "configs" / "default.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_default_config_loads_with_expected_types() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.seed == 42
    assert isinstance(cfg.data.img_size, int)
    assert isinstance(cfg.train.lr, float)
    assert cfg.train.lr == pytest.approx(1e-3)
    assert cfg.model.num_classes == 3
    assert len(cfg.data.mean) == len(cfg.data.std) == cfg.model.in_channels


def test_paths_are_resolved_against_repo_root() -> None:
    cfg = load_config()
    root = project_root()
    assert cfg.data.root == root / "data"
    assert cfg.paths.best_weights == root / "models" / "unet_pet_best.pt"
    assert all(p.is_absolute() for p in (cfg.paths.runs, cfg.paths.checkpoints))


def test_relative_config_path_works_from_other_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_config("configs/default.yaml").seed == 42


def test_overrides_convert_cli_strings() -> None:
    cfg = load_config(overrides={"train.epochs": "1", "train.lr": "5e-4", "seed": "7"})
    assert cfg.train.epochs == 1
    assert cfg.train.lr == pytest.approx(5e-4)
    assert cfg.seed == 7


def test_override_does_not_mutate_other_loads() -> None:
    load_config(overrides={"train.epochs": 1})
    assert load_config().train.epochs == 25


@pytest.mark.parametrize("key", ["train.epoch", "nope", "data.img_size.x"])
def test_unknown_override_key_raises(key: str) -> None:
    with pytest.raises(KeyError):
        load_config(overrides={key: 1})


def test_unknown_key_in_file_raises(tmp_path: Path, raw_default: dict) -> None:
    raw_default["train"]["lerning_rate"] = 0.1
    with pytest.raises(KeyError, match="lerning_rate"):
        load_config(_write_yaml(tmp_path, raw_default))


def test_missing_key_in_file_raises(tmp_path: Path, raw_default: dict) -> None:
    del raw_default["model"]["depth"]
    with pytest.raises(KeyError, match="depth"):
        load_config(_write_yaml(tmp_path, raw_default))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("data.val_fraction", 1.5),
        ("data.img_size", 100),  # not divisible by 2**depth = 16
        ("train.scheduler", "magic"),
        ("train.epochs", 0),
        ("data.mean", [0.5, 0.5]),
    ],
)
def test_invalid_values_raise(key: str, value: object) -> None:
    with pytest.raises(ValueError):
        load_config(overrides={key: value})


def test_unconvertible_value_raises() -> None:
    with pytest.raises(ValueError, match="train.epochs"):
        load_config(overrides={"train.epochs": "many"})


def test_to_dict_round_trips_through_yaml(tmp_path: Path) -> None:
    cfg = load_config()
    dumped = cfg.to_dict()
    assert "source" not in dumped
    reloaded = load_config(_write_yaml(tmp_path, dumped))
    assert reloaded == cfg
