"""Smoke tests for unet_pet.train: tiny model, fake data, everything in tmp_path (CPU)."""

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from unet_pet import train as train_module
from unet_pet.config import load_config
from unet_pet.model import UNet
from unet_pet.train import HISTORY_KEYS, fit, main


def _fake_split(n: int, seed: int) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(n, 3, 32, 32, generator=g)
    masks = torch.ones(n, 32, 32, dtype=torch.long)  # background
    masks[:, 8:24, 8:24] = 0  # pet
    masks[:, 8:24, 8] = 2  # border
    images[:, 0, 8:24, 8:24] += 2.0  # make the pet visible in the red channel
    return TensorDataset(images, masks)


def _fake_loaders(batch_size: int = 4) -> dict[str, DataLoader]:
    return {
        "train": DataLoader(_fake_split(12, 0), batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(0)),
        "val": DataLoader(_fake_split(6, 1), batch_size=batch_size),
        "test": DataLoader(_fake_split(4, 2), batch_size=batch_size),
    }  # fmt: skip


@pytest.fixture
def cfg(tmp_path: Path):
    return load_config(
        overrides={
            "data.img_size": 32,
            "train.epochs": 2,
            "train.batch_size": 4,
            "train.lr": 1e-2,
            "train.num_log_images": 2,
            "paths.runs": str(tmp_path / "runs"),
            "paths.checkpoints": str(tmp_path / "checkpoints"),
            "paths.best_weights": str(tmp_path / "models" / "best.pt"),
            "paths.history": str(tmp_path / "models" / "history.json"),
        }
    )


def _tiny_model() -> UNet:
    torch.manual_seed(0)  # identical initial weights in every call
    return UNet(base_channels=4, depth=2)


def _variant(cfg, overrides: dict):
    """The test config with a few values changed (everything else, incl. paths, kept)."""
    flat = {
        f"{section}.{key}": value
        for section, values in cfg.to_dict().items()
        if isinstance(values, dict)
        for key, value in values.items()
    }
    return load_config(cfg.source, overrides={**flat, **overrides})


def test_fit_writes_all_outputs(cfg) -> None:
    result = fit(cfg, model=_tiny_model(), loaders=_fake_loaders(),
                 device=torch.device("cpu"), run_name="t", progress=False)  # fmt: skip
    run_dir, ckpt_dir = result["run_dir"], result["checkpoint_dir"]
    assert run_dir.name.endswith("_t") and run_dir.parent == cfg.paths.runs
    assert list(run_dir.glob("events.out.tfevents.*")), "no TensorBoard event file"
    assert (run_dir / "config.yaml").is_file() and (run_dir / "history.json").is_file()
    assert (ckpt_dir / "last.pt").is_file() and (ckpt_dir / "best.pt").is_file()

    history = result["history"]
    assert set(history) == set(HISTORY_KEYS)
    assert all(len(v) == cfg.train.epochs for v in history.values())
    assert 1 <= result["best_epoch"] <= cfg.train.epochs
    assert result["best_val_mIoU"] == max(history["val_mIoU"])

    # exported deliverable: plain state_dict, loadable with weights_only=True
    state = torch.load(cfg.paths.best_weights, map_location="cpu", weights_only=True)
    model = _tiny_model()
    model.load_state_dict(state)
    exported = json.loads(cfg.paths.history.read_text(encoding="utf-8"))
    assert exported["val_mIoU"] == history["val_mIoU"]
    assert exported["best_epoch"] == result["best_epoch"]


def test_training_actually_learns(cfg) -> None:
    long_cfg = _variant(cfg, {"train.epochs": 8})
    result = fit(long_cfg, model=_tiny_model(), loaders=_fake_loaders(),
                 device=torch.device("cpu"), export=False, progress=False)  # fmt: skip
    h = result["history"]
    assert h["train_loss"][-1] < h["train_loss"][0]
    assert result["best_val_mIoU"] > 0.5


def test_limit_batches_and_no_export(cfg, monkeypatch) -> None:
    calls = {"n": 0}
    original = _tiny_model().forward.__func__

    def counting_forward(self, x):
        calls["n"] += 1
        return original(self, x)

    monkeypatch.setattr(UNet, "forward", counting_forward)
    fit(cfg, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
        limit_batches=1, export=False, progress=False)  # fmt: skip
    # per epoch: 1 train batch + 1 val batch + 1 forward for the TensorBoard image grid
    assert calls["n"] == cfg.train.epochs * 3
    assert not cfg.paths.best_weights.exists() and not cfg.paths.history.exists()


def test_resume_continues_up_to_total_epochs(cfg) -> None:
    first = fit(cfg, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
                export=False, progress=False)  # fmt: skip
    more = _variant(cfg, {"train.epochs": 3})
    last = first["checkpoint_dir"] / "last.pt"
    resumed = fit(more, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
                  resume=last, export=False, progress=False)  # fmt: skip
    assert resumed["run_dir"] == first["run_dir"]  # same run, same TensorBoard curves
    assert len(resumed["history"]["val_mIoU"]) == 3
    assert resumed["history"]["val_mIoU"][:2] == first["history"]["val_mIoU"]


def test_cli_smoke(cfg, monkeypatch, tmp_path: Path) -> None:
    """``main`` end-to-end with the data loading swapped for fake loaders."""
    monkeypatch.setattr(train_module, "make_dataloaders", lambda c: _fake_loaders())
    monkeypatch.setattr(train_module, "build_model", lambda c: _tiny_model())
    config_file = tmp_path / "cfg.yaml"
    config_file.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    result = main(["--config", str(config_file), "--epochs", "1", "--limit-batches", "2",
                   "--run-name", "smoke", "--device", "cpu",
                   "--set", "train.dice_weight=0"])  # fmt: skip
    assert len(result["history"]["val_mIoU"]) == 1
    assert list(result["run_dir"].glob("events.out.tfevents.*"))
    assert not cfg.paths.best_weights.exists()  # --limit-batches never exports
    saved = yaml.safe_load((result["run_dir"] / "config.yaml").read_text(encoding="utf-8"))
    assert saved["train"]["dice_weight"] == 0.0 and saved["train"]["epochs"] == 1


def test_bad_set_argument_is_rejected() -> None:
    with pytest.raises(Exception, match="key=value"):
        main(["--set", "train.epochs"])


def test_fit_with_limit_batches_never_exports_even_by_default(cfg) -> None:
    result = fit(cfg, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
                 limit_batches=1, progress=False)  # fmt: skip
    assert result["exported"] is False
    assert not cfg.paths.best_weights.exists() and not cfg.paths.history.exists()


def test_invalid_limit_batches_rejected(cfg) -> None:
    with pytest.raises(ValueError, match="limit_batches"):
        fit(cfg, model=_tiny_model(), loaders=_fake_loaders(), limit_batches=0, progress=False)
    with pytest.raises(SystemExit):
        main(["--limit-batches", "0"])


def test_resume_is_identical_to_uninterrupted_run(cfg) -> None:
    """2 epochs + resume for the 3rd must equal 3 epochs in one go (per-epoch seeding)."""
    three = _variant(cfg, {"train.epochs": 3})
    straight = fit(three, model=_tiny_model(), loaders=_fake_loaders(),
                   device=torch.device("cpu"), export=False, progress=False)  # fmt: skip
    first = fit(cfg, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
                export=False, progress=False)  # fmt: skip
    resumed = fit(three, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
                  resume=first["checkpoint_dir"] / "last.pt", export=False,
                  progress=False)  # fmt: skip
    for key in ("train_loss", "val_loss", "val_mIoU", "lr"):
        assert resumed["history"][key] == pytest.approx(straight["history"][key]), key


def test_resume_from_moved_checkpoint_exports_correct_best(cfg, tmp_path: Path) -> None:
    first = fit(cfg, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
                export=False, progress=False)  # fmt: skip
    moved = tmp_path / "elsewhere" / "last.pt"
    moved.parent.mkdir()
    (first["checkpoint_dir"] / "last.pt").rename(moved)
    (first["checkpoint_dir"] / "best.pt").unlink()  # only the moved last.pt is left

    three = _variant(cfg, {"train.epochs": 3})
    resumed = fit(three, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
                  resume=moved, progress=False)  # fmt: skip
    assert resumed["checkpoint_dir"] == moved.parent and resumed["exported"] is True
    exported = torch.load(three.paths.best_weights, map_location="cpu", weights_only=True)
    best_in_ckpt = torch.load(moved, map_location="cpu", weights_only=True)["best_model"]
    assert all(torch.equal(exported[k], best_in_ckpt[k]) for k in exported)


def test_resume_of_limited_run_is_not_exported(cfg) -> None:
    smoke = fit(cfg, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
                limit_batches=1, progress=False)  # fmt: skip
    three = _variant(cfg, {"train.epochs": 3})
    resumed = fit(three, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
                  resume=smoke["checkpoint_dir"] / "last.pt", progress=False)  # fmt: skip
    assert resumed["exported"] is False and not three.paths.best_weights.exists()


def test_resume_with_no_epochs_left_changes_nothing(cfg) -> None:
    first = fit(cfg, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
                export=False, progress=False)  # fmt: skip
    before = sorted(p.name for p in first["run_dir"].iterdir())
    again = fit(cfg, model=_tiny_model(), loaders=_fake_loaders(), device=torch.device("cpu"),
                resume=first["checkpoint_dir"] / "last.pt", progress=False)  # fmt: skip
    assert again["exported"] is False and not cfg.paths.best_weights.exists()
    assert sorted(p.name for p in first["run_dir"].iterdir()) == before
    assert again["history"] == first["history"]
