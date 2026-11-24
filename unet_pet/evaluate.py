"""Evaluation of a trained model on a data split, with a CLI entry point.

Usage::

    python -m unet_pet.evaluate --weights models/unet_pet_best.pt              # test split
    python -m unet_pet.evaluate --split sample                                 # offline demo
    python -m unet_pet.evaluate --split test --limit 50 --figures              # quick check

What is computed (all from the confusion matrix of the *whole* split, not batch means):

- model metrics: mIoU (main metric), macro Dice, pixel accuracy, per-class IoU / Dice;
- the same metrics for the "always background" baseline, so the numbers have a reference
  point (pixel accuracy alone looks good even for a model that never finds the pet);
- per-image mIoU, used to pick the best / worst predicted images for error analysis.

Outputs go to ``cfg.paths.results``: ``<split>_metrics.json`` (``test_metrics.json`` is the
committed result the README and the notebook quote) and, with ``--figures``, PNG figures.
A ``--limit`` run is only a quick check: it prints the table but writes no JSON, so it
can never overwrite the real result.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from unet_pet.config import DEFAULT_CONFIG_PATH, Config, load_config
from unet_pet.data import (
    CLASS_NAMES,
    PetSegmentationDataset,
    SampleFolder,
    build_transform,
    load_base_datasets,
    make_datasets,
)
from unet_pet.metrics import SegmentationMeter
from unet_pet.model import build_model
from unet_pet.utils import (
    get_device,
    parse_overrides,
    positive_int,
    project_root,
    resolve_path,
    set_seed,
)
from unet_pet.visualize import plot_confusion_matrix, show_predictions

SPLITS = ("test", "val", "sample")
BACKGROUND = CLASS_NAMES.index("background")


# --------------------------------------------------------------------------------------
# Model and data
# --------------------------------------------------------------------------------------


def load_trained_model(cfg: Config, weights: str | Path, device: torch.device) -> nn.Module:
    """Build the model described by ``cfg.model`` and load a ``state_dict`` into it.

    Raises:
        FileNotFoundError: the weights file does not exist.
        RuntimeError: the weights were saved from a different architecture.
    """
    path = resolve_path(weights)
    if not path.is_file():
        raise FileNotFoundError(
            f"Weights not found: {path}. Train first (python -m unet_pet.train) or pass --weights."
        )
    model = build_model(cfg)
    state = torch.load(path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{path} does not match the architecture in the config (model.*: "
            f"{cfg.model}). Use the config the weights were trained with."
        ) from exc
    return model.to(device).eval()


def make_eval_dataset(
    cfg: Config, split: str = "test", limit: int | None = None, download: bool = True
) -> PetSegmentationDataset:
    """The non-augmented dataset of ``split`` (``test``, ``val`` or ``sample``).

    ``limit`` keeps the first ``limit`` images (quick checks). ``sample`` is the
    ``sample_data/`` folder and needs no download.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
    if split == "sample":
        dataset = PetSegmentationDataset(SampleFolder(), build_transform(cfg, train=False))
    else:
        try:
            bases = load_base_datasets(cfg, download=download)
        except (OSError, RuntimeError) as exc:  # no internet / no local copy
            raise RuntimeError(
                f"Could not load the Oxford-IIIT Pet '{split}' split ({exc}). "
                "Without the dataset use --split sample."
            ) from exc
        dataset = make_datasets(cfg, *bases)[split]
    if limit is not None:
        dataset = PetSegmentationDataset(dataset.base, dataset.transform, dataset.indices[:limit])
    return dataset


def image_names(dataset: PetSegmentationDataset) -> list[str]:
    """File stem (e.g. ``"Abyssinian_201"``) of every image; the index if unknown."""
    base = dataset.base
    if hasattr(base, "names"):  # SampleFolder
        stems = list(base.names)
    elif hasattr(base, "_images"):  # torchvision OxfordIIITPet (list of image paths)
        stems = [Path(p).stem for p in base._images]
    else:
        return [str(i) for i in range(len(dataset))]
    return [stems[i] for i in dataset.indices]


def make_eval_loader(cfg: Config, dataset: PetSegmentationDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,  # per-image results must stay aligned with the dataset indices
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    progress: bool = True,
) -> dict[str, dict[str, Any]]:
    """Model and "always background" baseline metrics over a whole loader, in one pass.

    Returns:
        ``{"model": ..., "baseline": ...}``, each a ``SegmentationMeter.compute()`` dict.
        ``model["per_image_iou"][i]`` belongs to the ``i``-th image of the loader.
    """
    model.eval()
    model_meter, baseline_meter = SegmentationMeter(num_classes), SegmentationMeter(num_classes)
    for images, masks in tqdm(loader, desc="evaluate", leave=False, disable=not progress):
        logits = model(images.to(device))
        model_meter.update(logits.float().cpu(), masks)
        baseline_meter.update(torch.full_like(masks, BACKGROUND), masks)
    baseline = baseline_meter.compute()
    baseline.pop("per_image_iou")
    return {"model": model_meter.compute(), "baseline": baseline}


def select_examples(per_image_iou: Sequence[float], k: int = 3) -> dict[str, list[int]]:
    """Indices of the ``k`` best (highest IoU first) and ``k`` worst (lowest first) images.

    NaN values are skipped. With fewer than ``2k`` images the two lists may overlap.
    """
    valid = [(iou, i) for i, iou in enumerate(per_image_iou) if not math.isnan(iou)]
    ranked = sorted(valid)
    return {
        "best": [i for _, i in reversed(ranked[-k:])] if k > 0 else [],
        "worst": [i for _, i in ranked[:k]],
    }


def _json_safe(value: Any) -> Any:
    """Replace NaN (invalid in strict JSON) with ``None``, recursively."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Mapping):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def build_report(
    results: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
    *,
    split: str,
    weights: str | Path,
    k: int = 3,
) -> dict[str, Any]:
    """JSON-ready summary: metrics of model and baseline, best / worst images, per image."""
    model = dict(results["model"])
    per_image = model.pop("per_image_iou")
    examples = select_examples(per_image, k)
    return {
        "split": split,
        "n_images": len(per_image),
        "weights": _display_path(resolve_path(weights)),
        "class_names": list(CLASS_NAMES),
        "model": model,
        "baseline": dict(results["baseline"]),
        "examples": {
            kind: [{"index": i, "name": names[i], "iou": per_image[i]} for i in indices]
            for kind, indices in examples.items()
        },
        "per_image_iou": {"names": list(names), "iou": list(per_image)},
    }


def format_report(report: Mapping[str, Any]) -> str:
    """Plain-text table of the model vs. the baseline (for the console)."""
    m, b = report["model"], report["baseline"]
    rows = [("mIoU", m["mIoU"], b["mIoU"]), ("Dice (macro)", m["dice"], b["dice"]),
            ("pixel accuracy", m["pixel_acc"], b["pixel_acc"])]  # fmt: skip
    rows += [
        (f"IoU {name}", mi, bi)
        for name, mi, bi in zip(
            report["class_names"], m["iou_per_class"], b["iou_per_class"], strict=True
        )  # fmt: skip
    ]

    def num(v: float | None) -> str:
        return "   n/a" if v is None or math.isnan(v) else f"{v:6.4f}"

    lines = [
        f"Split '{report['split']}', {report['n_images']} images, weights {report['weights']}",
        f"{'metric':<16} {'model':>8} {'baseline':>9}",
        *(f"{name:<16} {num(mv):>8} {num(bv):>9}" for name, mv, bv in rows),
    ]
    for kind in ("best", "worst"):
        items = ", ".join(f"{e['name']} ({e['iou']:.3f})" for e in report["examples"][kind])
        lines.append(f"{kind:<6} images: {items}")
    return "\n".join(lines)


def save_report(report: Mapping[str, Any], path: str | Path) -> Path:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(report), indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


@torch.no_grad()
def examples_figure(
    model: nn.Module,
    dataset: PetSegmentationDataset,
    report: Mapping[str, Any],
    cfg: Config,
    device: torch.device,
) -> Figure:
    """Best images first, then the worst ones: image | true mask | prediction | overlay."""
    entries = [("best", e) for e in report["examples"]["best"]]
    entries += [("worst", e) for e in report["examples"]["worst"]]
    images, masks = zip(*(dataset[e["index"]] for _, e in entries), strict=True)
    images_t, masks_t = torch.stack(images), torch.stack(masks)
    preds = model.eval()(images_t.to(device)).argmax(dim=1).cpu()
    titles = [f"{kind}: {e['name']}\nIoU {e['iou']:.3f}" for kind, e in entries]
    return show_predictions(images_t, masks_t, preds, mean=cfg.data.mean, std=cfg.data.std,
                            n=len(entries), titles=titles)  # fmt: skip


def save_figures(
    model: nn.Module,
    dataset: PetSegmentationDataset,
    report: Mapping[str, Any],
    cfg: Config,
    device: torch.device,
    out_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    split = report["split"]
    figures = {
        out_dir / f"{split}_confusion_matrix.png": plot_confusion_matrix(
            report["model"]["confusion_matrix"], title=f"Confusion matrix ({split})"
        ),
        out_dir / f"{split}_best_worst.png": examples_figure(model, dataset, report, cfg, device),
    }
    for path, fig in figures.items():
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return list(figures)


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------


def run_evaluation(
    cfg: Config,
    weights: str | Path | None = None,
    *,
    split: str = "test",
    limit: int | None = None,
    device: torch.device | None = None,
    figures: bool = False,
    download: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate ``weights`` (default: ``cfg.paths.best_weights``) on ``split``.

    Writes ``<split>_metrics.json`` into ``cfg.paths.results`` unless ``limit`` is set.

    Returns:
        ``{"report", "json_path" (None for limited runs), "figure_paths"}``.
    """
    set_seed(cfg.seed)
    device = get_device() if device is None else device
    weights = cfg.paths.best_weights if weights is None else weights
    model = load_trained_model(cfg, weights, device)
    dataset = make_eval_dataset(cfg, split, limit=limit, download=download)
    results = evaluate(model, make_eval_loader(cfg, dataset), device, cfg.model.num_classes,
                       progress=progress)  # fmt: skip
    report = build_report(results, image_names(dataset), split=split, weights=weights)

    json_path = None
    if limit is None:
        json_path = save_report(report, cfg.paths.results / f"{split}_metrics.json")
    figure_paths = (
        save_figures(model, dataset, report, cfg, device, cfg.paths.results) if figures else []
    )
    return {"report": report, "json_path": json_path, "figure_paths": figure_paths}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """CLI entry point (``python -m unet_pet.evaluate``)."""
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config file")
    parser.add_argument("--weights", help="state_dict file (default: paths.best_weights)")
    parser.add_argument("--split", choices=SPLITS, default="test", help="data to evaluate on")
    parser.add_argument(
        "--limit", type=positive_int, help="only the first N images (quick check; no JSON)"
    )
    parser.add_argument("--figures", action="store_true", help="save PNG figures to results/")
    parser.add_argument("--device", help="force a device, e.g. cpu (default: auto)")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="extra config overrides"
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    out = run_evaluation(cfg, args.weights, split=args.split, limit=args.limit,
                         device=get_device(args.device), figures=args.figures)  # fmt: skip
    print(format_report(out["report"]))
    if out["json_path"] is not None:
        print(f"Saved {_display_path(out['json_path'])}")
    else:
        print("Limited run (--limit): no JSON written.")
    for path in out["figure_paths"]:
        print(f"Saved {_display_path(path)}")
    return out


if __name__ == "__main__":
    main()
