"""Training loop, checkpointing and TensorBoard logging, with a CLI entry point.

Usage::

    python -m unet_pet.train --config configs/default.yaml
    python -m unet_pet.train --epochs 1 --limit-batches 2 --run-name smoke   # quick check
    python -m unet_pet.train --resume checkpoints/<run>/last.pt              # continue

Outputs of a run named ``<stamp>_<name>``:

- ``runs/<stamp>_<name>/``: TensorBoard events, ``config.yaml``, ``history.json``
- ``checkpoints/<stamp>_<name>/last.pt``: full state (model, optimiser, scheduler, history)
  to resume from, and ``best.pt``: weights of the best validation-mIoU epoch
- ``models/unet_pet_best.pt`` (``state_dict`` only) and ``models/unet_pet_history.json``:
  the exported deliverable. Not written for quick runs (``--limit-batches``) or with
  ``--no-export``, so a smoke test can never overwrite the committed weights.

Model selection uses the *validation* mIoU only; the test split is never touched here.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from unet_pet.config import DEFAULT_CONFIG_PATH, Config, load_config
from unet_pet.data import CLASS_NAMES, make_dataloaders
from unet_pet.losses import CombinedLoss, build_loss
from unet_pet.metrics import SegmentationMeter
from unet_pet.model import build_model, count_parameters
from unet_pet.utils import get_device, parse_overrides, positive_int, resolve_path, set_seed
from unet_pet.visualize import make_tb_grid

HISTORY_KEYS = (
    "train_loss", "train_ce", "train_dice",
    "val_loss", "val_ce", "val_dice_loss",
    "val_mIoU", "val_dice", "val_pixel_acc", "lr",
)  # fmt: skip


def _autocast(device: torch.device, enabled: bool):
    """Mixed precision only on CUDA (on CPU/MPS it would be slower or unsupported)."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CombinedLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
    limit_batches: int | None = None,
    progress: bool = True,
    desc: str = "train",
) -> dict[str, float]:
    """One pass over ``loader``; returns the pixel-weighted mean ``loss``, ``ce``, ``dice``."""
    model.train()
    totals = {"loss": 0.0, "ce": 0.0, "dice": 0.0}
    seen = 0
    n_batches = len(loader) if limit_batches is None else min(limit_batches, len(loader))
    bar = tqdm(loader, total=n_batches, desc=desc, leave=False, disable=not progress)
    for step, (images, masks) in enumerate(bar):
        if step >= n_batches:
            break
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            loss = loss_fn(model(images), masks)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        b = images.shape[0]
        seen += b
        totals["loss"] += loss.item() * b
        totals["ce"] += loss_fn.last_parts["ce"] * b
        totals["dice"] += loss_fn.last_parts["dice"] * b
        bar.set_postfix(loss=f"{loss.item():.3f}")
    return {k: v / max(seen, 1) for k, v in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CombinedLoss,
    device: torch.device,
    num_classes: int,
    *,
    amp: bool = False,
    limit_batches: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Loss and split-level metrics (``mIoU``, ``dice``, ``pixel_acc``, per class, cm)."""
    model.eval()
    meter = SegmentationMeter(num_classes)
    totals = {"loss": 0.0, "ce": 0.0, "dice_loss": 0.0}
    seen = 0
    n_batches = len(loader) if limit_batches is None else min(limit_batches, len(loader))
    for step, (images, masks) in enumerate(
        tqdm(loader, total=n_batches, desc="val", leave=False, disable=not progress)
    ):
        if step >= n_batches:
            break
        images, masks = images.to(device), masks.to(device)
        with _autocast(device, amp):
            logits = model(images)
            loss = loss_fn(logits, masks)
        meter.update(logits.float(), masks)
        b = images.shape[0]
        seen += b
        totals["loss"] += loss.item() * b
        totals["ce"] += loss_fn.last_parts["ce"] * b
        totals["dice_loss"] += loss_fn.last_parts["dice"] * b
    results = meter.compute()
    results.pop("per_image_iou")
    return {**{k: v / max(seen, 1) for k, v in totals.items()}, **results}


def _build_scheduler(cfg: Config, optimizer: torch.optim.Optimizer):
    if cfg.train.scheduler == "reduce_on_plateau":
        # mode="max": the monitored value is the validation mIoU, higher is better.
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.train.scheduler_factor,
            patience=cfg.train.scheduler_patience,
        )
    return None


def _fixed_val_batch(loader: DataLoader, n: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The first ``n`` validation samples: the same images every epoch, so progress is visible."""
    if n <= 0 or len(loader.dataset) == 0:
        return None
    items = [loader.dataset[i] for i in range(min(n, len(loader.dataset)))]
    return torch.stack([x for x, _ in items]), torch.stack([y for _, y in items])


def _flat_hparams(cfg: Config) -> dict[str, Any]:
    """Scalar config values for TensorBoard's HPARAMS tab (lists/paths as strings)."""
    flat: dict[str, Any] = {"seed": cfg.seed}
    for section in ("data", "model", "train"):
        for key, value in cfg.to_dict()[section].items():
            flat[f"{section}.{key}"] = (
                value if isinstance(value, (int, float, str, bool)) else str(value)
            )
    return flat


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _seed_epoch(seed: int, epoch: int, train_loader: DataLoader) -> None:
    """Re-seed every RNG (and the train loader's shuffling) from ``seed + epoch``.

    Epoch ``k`` then always sees the same shuffling and augmentation, whether the run went
    straight through or was resumed from a checkpoint at epoch ``k - 1``.
    """
    set_seed(seed + epoch)
    generator = getattr(train_loader, "generator", None)
    if generator is not None:  # RandomSampler and worker seeds draw from this generator
        generator.manual_seed(seed + epoch)


def _config_differences(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Differing ``section.key`` values, ignoring ``train.epochs`` and the ``paths``."""
    diffs = []
    for section, values in new.items():
        if section == "paths":
            continue
        if not isinstance(values, Mapping):
            if old.get(section) != values:
                diffs.append(f"{section}: {old.get(section)!r} -> {values!r}")
            continue
        for key, value in values.items():
            before = old.get(section, {}).get(key)
            if (section, key) != ("train", "epochs") and before != value:
                diffs.append(f"{section}.{key}: {before!r} -> {value!r}")
    return diffs


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit(
    cfg: Config,
    *,
    model: nn.Module | None = None,
    loaders: Mapping[str, DataLoader] | None = None,
    device: torch.device | None = None,
    run_name: str = "",
    limit_batches: int | None = None,
    export: bool = True,
    resume: str | Path | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Train ``cfg.train.epochs`` epochs with validation after each one.

    Args:
        model / loaders / device: injectable (tests use a tiny model and fake data);
            built from ``cfg`` when omitted.
        run_name: suffix of the run directory name.
        limit_batches: use only this many batches per epoch and split (quick checks).
            A limited run never exports, so it cannot overwrite the deliverable.
        export: write the best weights and history to ``cfg.paths`` (the deliverable).
        resume: path of a ``last.pt`` checkpoint to continue from; training then runs up
            to ``cfg.train.epochs`` epochs in total, in the same run and checkpoint folder.

    Returns:
        ``{"history", "best_val_mIoU", "best_epoch", "run_dir", "checkpoint_dir", "exported"}``.
    """
    if limit_batches is not None and limit_batches < 1:
        raise ValueError("limit_batches must be >= 1")
    export = export and limit_batches is None

    set_seed(cfg.seed)
    device = get_device() if device is None else device
    model = (build_model(cfg) if model is None else model).to(device)
    loaders = make_dataloaders(cfg) if loaders is None else loaders
    loss_fn = build_loss(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = _build_scheduler(cfg, optimizer)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    config_dict = cfg.to_dict()

    history: dict[str, list[float]] = {k: [] for k in HISTORY_KEYS}
    best_miou, best_epoch, start_epoch = -1.0, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    if resume is not None:
        resume_path = resolve_path(resume)
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        history = {k: list(ckpt["history"].get(k, [])) for k in HISTORY_KEYS}
        best_miou, best_epoch = ckpt["best_val_mIoU"], ckpt["best_epoch"]
        start_epoch, run_id = ckpt["epoch"], ckpt["run_id"]
        best_state = ckpt.get("best_model")
        ckpt_dir = resume_path.parent  # keep writing next to the checkpoint we resumed from
        if ckpt.get("limit_batches") is not None and export:
            export = False
            print(
                f"Warning: {resume_path} comes from a --limit-batches run; "
                "this run will not be exported."
            )
        diffs = _config_differences(ckpt.get("config", {}), config_dict)
        if diffs:
            print("Warning: config differs from the checkpoint: " + "; ".join(diffs))
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S") + (f"_{run_name}" if run_name else "")
        ckpt_dir = cfg.paths.checkpoints / run_id
    run_dir = cfg.paths.runs / run_id
    result: dict[str, Any] = {"run_dir": run_dir, "checkpoint_dir": ckpt_dir, "exported": False}

    if start_epoch >= cfg.train.epochs:
        print(
            f"Nothing to do: the checkpoint is already at epoch {start_epoch} and "
            f"train.epochs = {cfg.train.epochs}. Increase --epochs to train further."
        )
        return {**result, "history": history, "best_val_mIoU": best_miou, "best_epoch": best_epoch}

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_name = "config.yaml" if start_epoch == 0 else f"config_resume_epoch{start_epoch}.yaml"
    (run_dir / config_name).write_text(yaml.safe_dump(config_dict, sort_keys=False), "utf-8")

    writer = SummaryWriter(log_dir=str(run_dir))
    fixed_batch = _fixed_val_batch(loaders["val"], cfg.train.num_log_images)
    if progress:
        print(
            f"Run {run_id} | device {device} | AMP {use_amp} | "
            f"{count_parameters(model):,} parameters | "
            f"train {len(loaders['train'].dataset)} / val {len(loaders['val'].dataset)} images"
            + (f" | resuming after epoch {start_epoch}" if start_epoch else "")
        )
    try:
        for epoch in range(start_epoch + 1, cfg.train.epochs + 1):
            _seed_epoch(cfg.seed, epoch, loaders["train"])
            lr = optimizer.param_groups[0]["lr"]
            tr = train_one_epoch(
                model, loaders["train"], loss_fn, optimizer, device, scaler=scaler,
                amp=use_amp, limit_batches=limit_batches, progress=progress,
                desc=f"epoch {epoch}/{cfg.train.epochs}",
            )  # fmt: skip
            va = validate(
                model, loaders["val"], loss_fn, device, cfg.model.num_classes,
                amp=use_amp, limit_batches=limit_batches, progress=progress,
            )  # fmt: skip
            if scheduler is not None:
                scheduler.step(va["mIoU"])

            row = {
                "train_loss": tr["loss"], "train_ce": tr["ce"], "train_dice": tr["dice"],
                "val_loss": va["loss"], "val_ce": va["ce"], "val_dice_loss": va["dice_loss"],
                "val_mIoU": va["mIoU"], "val_dice": va["dice"], "val_pixel_acc": va["pixel_acc"],
                "lr": lr,
            }  # fmt: skip
            for key, value in row.items():
                history[key].append(value)

            writer.add_scalar("loss/train", tr["loss"], epoch)
            writer.add_scalar("loss/val", va["loss"], epoch)
            writer.add_scalar("loss/train_ce", tr["ce"], epoch)
            writer.add_scalar("loss/train_dice", tr["dice"], epoch)
            writer.add_scalar("metrics/val_mIoU", va["mIoU"], epoch)
            writer.add_scalar("metrics/val_dice", va["dice"], epoch)
            writer.add_scalar("metrics/val_pixel_acc", va["pixel_acc"], epoch)
            for name, iou in zip(CLASS_NAMES, va["iou_per_class"], strict=False):
                writer.add_scalar(f"metrics/val_IoU_{name}", iou, epoch)
            writer.add_scalar("lr", lr, epoch)
            every = cfg.train.log_images_every
            if fixed_batch is not None and every > 0 and epoch % every == 0:
                model.eval()
                with torch.no_grad():
                    preds = model(fixed_batch[0].to(device)).argmax(dim=1).cpu()
                grid = make_tb_grid(
                    fixed_batch[0], fixed_batch[1], preds,
                    mean=cfg.data.mean, std=cfg.data.std, n=cfg.train.num_log_images,
                )  # fmt: skip
                writer.add_image("images/val_predictions", grid, epoch)

            improved = va["mIoU"] > best_miou
            if improved:
                best_miou, best_epoch = va["mIoU"], epoch
                best_state = _cpu_state(model)
                torch.save(best_state, ckpt_dir / "best.pt")
            # last.pt is self-contained (incl. the best weights), so it can be moved
            # elsewhere and still be resumed and exported correctly.
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "epoch": epoch,
                    "history": history,
                    "best_val_mIoU": best_miou,
                    "best_epoch": best_epoch,
                    "best_model": best_state,
                    "run_id": run_id,
                    "limit_batches": limit_batches,
                    "config": config_dict,
                },
                ckpt_dir / "last.pt",
            )
            _write_json(run_dir / "history.json", history)
            writer.flush()
            if progress:
                print(
                    f"epoch {epoch:3d} | train loss {tr['loss']:.4f} | val loss {va['loss']:.4f}"
                    f" | val mIoU {va['mIoU']:.4f} | lr {lr:.2e}" + ("  *best*" if improved else "")
                )

        writer.add_hparams(_flat_hparams(cfg), {"hparam/best_val_mIoU": best_miou}, run_name=".")
    finally:
        writer.close()

    if export and best_state is not None:
        best_weights = cfg.paths.best_weights
        best_weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, best_weights)
        _write_json(cfg.paths.history, {**history, "best_epoch": best_epoch, "run_id": run_id})
        result["exported"] = True
        if progress:
            print(
                f"Exported best weights (epoch {best_epoch}, val mIoU {best_miou:.4f})"
                f" to {best_weights}"
            )
    elif progress:
        print(f"Not exported; best weights of this run: {ckpt_dir / 'best.pt'}")

    return {**result, "history": history, "best_val_mIoU": best_miou, "best_epoch": best_epoch}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point (``python -m unet_pet.train``)."""
    parser = argparse.ArgumentParser(description="Train the U-Net on Oxford-IIIT Pet.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config file")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument(
        "--limit-batches", type=positive_int, help="batches per epoch (quick check; no export)"
    )
    parser.add_argument("--run-name", default="", help="suffix of the run directory")
    parser.add_argument("--resume", help="continue from a checkpoints/<run>/last.pt file")
    parser.add_argument("--no-export", action="store_true", help="do not write models/")
    parser.add_argument("--device", help="force a device, e.g. cpu (default: auto)")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="extra config overrides, e.g. --set train.dice_weight=0",
    )
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = parse_overrides(args.set)
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    cfg = load_config(args.config, overrides=overrides)
    return fit(
        cfg,
        device=get_device(args.device),
        run_name=args.run_name,
        limit_batches=args.limit_batches,
        export=not args.no_export,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
