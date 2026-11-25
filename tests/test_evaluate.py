"""Tests for unet_pet.evaluate: CPU only, no download (fake loaders or sample_data/)."""

import json
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from unet_pet.config import load_config
from unet_pet.evaluate import (
    BACKGROUND,
    build_report,
    evaluate,
    format_report,
    image_names,
    load_trained_model,
    main,
    make_eval_dataset,
    metrics_path,
    run_evaluation,
    save_report,
    select_examples,
)
from unet_pet.metrics import SegmentationMeter
from unet_pet.model import UNet

TINY = {"model.base_channels": 4, "model.depth": 2, "data.img_size": 32}


@pytest.fixture
def cfg(tmp_path: Path):
    return load_config(overrides={**TINY, "train.batch_size": 4,
                                  "paths.results": str(tmp_path / "results")})  # fmt: skip


@pytest.fixture
def weights(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    path = tmp_path / "tiny.pt"
    torch.save(UNet(base_channels=4, depth=2).state_dict(), path)
    return path


class _FixedLogits(torch.nn.Module):
    """'Model' that returns given logits batch by batch (predictions fully under control)."""

    def __init__(self, preds: torch.Tensor, num_classes: int = 3) -> None:
        super().__init__()
        self.logits = torch.nn.functional.one_hot(preds, num_classes).permute(0, 3, 1, 2).float()
        self.pos = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.logits[self.pos : self.pos + x.shape[0]]
        self.pos += x.shape[0]
        return out


def test_baseline_matches_hand_computation() -> None:
    # 1 image, 4 pixels: pet, pet, background, border
    masks = torch.tensor([[[0, 0], [1, 2]]])
    loader = DataLoader(TensorDataset(torch.zeros(1, 3, 2, 2), masks), batch_size=1)
    out = evaluate(_FixedLogits(masks.clone()), loader, torch.device("cpu"), 3, progress=False)
    base = out["baseline"]
    # always background: IoU pet 0/2 = 0, background 1/4, border 0/1 = 0
    assert base["iou_per_class"] == pytest.approx([0.0, 0.25, 0.0])
    assert base["mIoU"] == pytest.approx(0.25 / 3)
    assert base["pixel_acc"] == pytest.approx(0.25)
    assert out["model"]["mIoU"] == pytest.approx(1.0)  # perfect "model"
    assert BACKGROUND == 1


def test_model_metrics_equal_segmentation_meter() -> None:
    g = torch.Generator().manual_seed(0)
    masks = torch.randint(0, 3, (6, 8, 8), generator=g)
    preds = torch.randint(0, 3, (6, 8, 8), generator=g)
    loader = DataLoader(TensorDataset(torch.zeros(6, 3, 8, 8), masks), batch_size=4)
    out = evaluate(_FixedLogits(preds), loader, torch.device("cpu"), 3, progress=False)
    meter = SegmentationMeter(3)
    meter.update(preds, masks)
    expected = meter.compute()
    assert out["model"]["confusion_matrix"] == expected["confusion_matrix"]
    assert out["model"]["mIoU"] == pytest.approx(expected["mIoU"])
    assert out["model"]["per_image_iou"] == pytest.approx(expected["per_image_iou"])
    assert "per_image_iou" not in out["baseline"]


def test_select_examples_orders_and_skips_nan() -> None:
    ious = [0.5, float("nan"), 0.9, 0.1, 0.7, 0.3]
    assert select_examples(ious, k=2) == {"best": [2, 4], "worst": [3, 5]}
    assert select_examples([0.4], k=3) == {"best": [0], "worst": [0]}
    assert select_examples([float("nan")], k=3) == {"best": [], "worst": []}


def test_report_json_is_strict_and_complete(tmp_path: Path) -> None:
    results = {
        "model": {"mIoU": 0.5, "dice": 0.6, "pixel_acc": 0.7,
                  "iou_per_class": [0.4, 0.6, float("nan")],
                  "dice_per_class": [0.5, 0.7, float("nan")], "confusion_matrix": [[1, 0, 0]] * 3,
                  "per_image_iou": [0.2, 0.8]},
        "baseline": {"mIoU": 0.2, "dice": 0.3, "pixel_acc": 0.6, "iou_per_class": [0.0, 0.6, 0.0],
                     "dice_per_class": [0.0, 0.7, 0.0], "confusion_matrix": [[0, 1, 0]] * 3},
    }  # fmt: skip
    report = build_report(results, ["a", "b"], split="test", weights=tmp_path / "w.pt", k=1)
    assert report["n_images"] == 2
    assert report["examples"] == {"best": [{"index": 1, "name": "b", "iou": 0.8}],
                                  "worst": [{"index": 0, "name": "a", "iou": 0.2}]}  # fmt: skip
    path = save_report(report, tmp_path / "out" / "test_metrics.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))  # strict: NaN would not parse back
    assert loaded["model"]["iou_per_class"][2] is None
    assert "n/a" in format_report(report)


def test_load_trained_model_errors(cfg, weights: Path, tmp_path: Path) -> None:
    model = load_trained_model(cfg, weights, torch.device("cpu"))
    assert not model.training
    with pytest.raises(FileNotFoundError, match="Weights not found"):
        load_trained_model(cfg, tmp_path / "missing.pt", torch.device("cpu"))
    wrong = load_config(overrides={**TINY, "model.base_channels": 8})
    with pytest.raises(RuntimeError, match="does not match the architecture"):
        load_trained_model(wrong, weights, torch.device("cpu"))


def test_sample_split_dataset_and_names(cfg) -> None:
    dataset = make_eval_dataset(cfg, "sample", limit=3)
    assert len(dataset) == 3
    names = image_names(dataset)
    assert len(names) == 3 and all(n and not n.isdigit() for n in names)
    with pytest.raises(ValueError, match="split"):
        make_eval_dataset(cfg, "train")


def test_run_evaluation_on_sample_writes_json_and_figures(cfg, weights: Path) -> None:
    out = run_evaluation(cfg, weights, split="sample", device=torch.device("cpu"),
                         figures=True, progress=False)  # fmt: skip
    report = out["report"]
    assert out["json_path"] == cfg.paths.results / "sample_metrics.json"
    saved = json.loads(out["json_path"].read_text(encoding="utf-8"))
    assert saved["n_images"] == 8 and saved["split"] == "sample"
    assert saved["model"]["mIoU"] == pytest.approx(report["model"]["mIoU"])
    assert 0.0 <= report["model"]["mIoU"] <= 1.0
    assert all(p.is_file() and p.suffix == ".png" for p in out["figure_paths"])


def test_cli_with_limit_prints_but_writes_nothing(cfg, weights: Path, capsys) -> None:
    """Acceptance criterion: the CLI runs with fake weights and --limit 2."""
    out = main(["--weights", str(weights), "--split", "sample", "--limit", "2", "--device",
                "cpu", "--set", *(f"{k}={v}" for k, v in TINY.items()),
                f"paths.results={cfg.paths.results}"])  # fmt: skip
    assert out["report"]["n_images"] == 2 and out["json_path"] is None
    assert not cfg.paths.results.exists()
    printed = capsys.readouterr().out
    assert "mIoU" in printed and "baseline" in printed and "no JSON written" in printed
    assert not math.isnan(out["report"]["baseline"]["mIoU"])
    with pytest.raises(SystemExit):
        main(["--limit", "0"])


def test_empty_pet_masks_are_reported() -> None:
    masks = torch.ones(3, 4, 4, dtype=torch.long)  # all background ...
    masks[1, :2] = 0  # ... except image 1, which has pet pixels
    loader = DataLoader(TensorDataset(torch.zeros(3, 3, 4, 4), masks), batch_size=2)
    preds = torch.zeros_like(masks)  # the "model" sees a pet everywhere
    results = evaluate(_FixedLogits(preds), loader, torch.device("cpu"), 3, progress=False)
    assert results["pet_pixels"] == [0, 8, 0]
    report = build_report(results, ["a", "b", "c"], split="test", weights="w.pt")
    assert [e["name"] for e in report["empty_pet_masks"]] == ["a", "c"]
    assert "2 true mask(s) without any pet pixel" in format_report(report)


def test_tag_writes_separate_json(cfg, weights: Path) -> None:
    main_run = run_evaluation(cfg, weights, split="sample", device=torch.device("cpu"),
                              progress=False)  # fmt: skip
    tagged = run_evaluation(cfg, weights, split="sample", device=torch.device("cpu"),
                            tag="ce_only", progress=False)  # fmt: skip
    assert main_run["json_path"] == metrics_path(cfg, "sample")
    assert tagged["json_path"] == cfg.paths.results / "sample_metrics_ce_only.json"
    assert json.loads(tagged["json_path"].read_text(encoding="utf-8"))["tag"] == "ce_only"
    assert main_run["json_path"].is_file()  # untouched by the tagged run
    with pytest.raises(ValueError, match="tag"):
        run_evaluation(cfg, weights, split="sample", tag="../x", progress=False)
