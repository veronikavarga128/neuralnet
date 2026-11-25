"""Tests for unet_pet.visualize on fake data (Agg backend is set in conftest.py)."""

import matplotlib.pyplot as plt
import pytest
import torch
from matplotlib.figure import Figure

from unet_pet.data import denormalize
from unet_pet.visualize import (
    CLASS_COLORS,
    make_tb_grid,
    mask_to_rgb,
    overlay_mask,
    plot_confusion_matrix,
    plot_history,
    show_predictions,
)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    images = torch.randn(3, 3, 32, 32)
    masks = torch.randint(0, 3, (3, 32, 32))
    logits = torch.randn(3, 3, 32, 32)
    return images, masks, logits


def test_mask_to_rgb_uses_class_colors() -> None:
    rgb = mask_to_rgb(torch.tensor([[0, 1, 2]]))
    assert rgb.shape == (3, 1, 3)
    for cls, color in enumerate(CLASS_COLORS):
        assert rgb[:, 0, cls].tolist() == pytest.approx(list(color))


def test_overlay_keeps_background_and_tints_pet() -> None:
    image = torch.full((3, 2, 2), 0.5)
    mask = torch.tensor([[1, 0], [2, 1]])
    out = overlay_mask(image, mask, alpha=0.5)
    assert out.shape == image.shape
    assert torch.equal(out[:, 0, 0], image[:, 0, 0])  # background untouched
    expected_pet = 0.5 * image[:, 0, 1] + 0.5 * torch.tensor(CLASS_COLORS[0])
    assert torch.allclose(out[:, 0, 1], expected_pet)
    assert 0.0 <= out.min() and out.max() <= 1.0


@pytest.mark.parametrize("with_preds", [True, False])
def test_show_predictions_layout(batch, with_preds: bool) -> None:
    images, masks, logits = batch
    fig = show_predictions(
        images,
        masks,
        logits if with_preds else None,
        mean=MEAN,
        std=STD,
        n=2,
        titles=["a", "b", "c"],
    )
    assert isinstance(fig, Figure)
    columns = 4 if with_preds else 3
    assert len(fig.axes) == 2 * columns
    assert fig.axes[0].get_ylabel() == "a"


def test_show_predictions_accepts_label_preds(batch) -> None:
    images, masks, logits = batch
    fig = show_predictions(images, masks, logits.argmax(1), mean=MEAN, std=STD, n=10)
    assert len(fig.axes) == 3 * 4  # n is capped at the batch size


def test_make_tb_grid_shape_and_range(batch) -> None:
    images, masks, logits = batch
    grid = make_tb_grid(images, masks, logits, mean=MEAN, std=STD, n=2)
    # 2 rows x 4 tiles of 32 px, padding 2 around every tile
    assert grid.shape == (3, 2 * 34 + 2, 4 * 34 + 2)
    assert grid.min() >= 0.0 and grid.max() <= 1.0


def test_grid_first_tile_is_denormalized_image(batch) -> None:
    images, masks, logits = batch
    grid = make_tb_grid(images, masks, logits, mean=MEAN, std=STD, n=1)
    first_tile = grid[:, 2:34, 2:34]
    assert torch.allclose(first_tile, denormalize(images[0], MEAN, STD))


def test_plot_confusion_matrix_normalized_and_raw() -> None:
    cm = [[8, 2, 0], [1, 9, 0], [0, 0, 0]]  # empty last row must not give NaN
    fig = plot_confusion_matrix(cm)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts[:2] == ["0.80", "0.20"] and "nan" not in " ".join(texts)
    raw = plot_confusion_matrix(torch.tensor(cm), normalize=False)
    assert raw.axes[0].texts[0].get_text() == "8"


def test_plot_history_skips_missing_keys() -> None:
    history = {"train_loss": [1.0, 0.8, 0.6], "val_loss": [1.1, 0.9, 0.8], "lr": [1e-3] * 3}
    fig = plot_history(history)
    assert len(fig.axes) == 2  # loss + lr, no metric panel
    full = plot_history({**history, "val_mIoU": [0.5, 0.6, 0.7]})
    assert len(full.axes) == 3


def test_plot_history_accepts_exported_history_with_extra_fields() -> None:
    """models/unet_pet_history.json also holds best_epoch (int) and run_id (str)."""
    history = {"train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9], "val_mIoU": [0.5, 0.6],
               "lr": [1e-3] * 2, "best_epoch": 2, "run_id": "20261124-132319_ce_dice"}  # fmt: skip
    fig = plot_history(history)
    assert len(fig.axes) == 3
    assert list(fig.axes[0].lines[0].get_xdata()) == [1, 2]
