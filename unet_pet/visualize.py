"""Plotting helpers: prediction overlays, image grids, confusion matrix and history figures.

Image inputs are *normalised* model inputs ``(3, H, W)`` / ``(B, 3, H, W)``; every function
takes the normalisation ``mean`` / ``std`` and undoes it for display. Masks and predictions
are label tensors with the project convention ``0 = pet, 1 = background, 2 = border``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from torchvision.utils import make_grid

from unet_pet.data import CLASS_NAMES, denormalize

# RGB in [0, 1] per class: pet = orange, background = dark grey, border = yellow.
CLASS_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.5, 0.05),
    (0.2, 0.2, 0.2),
    (1.0, 0.9, 0.1),
)
BACKGROUND_CLASS = 1


def mask_to_rgb(
    mask: torch.Tensor, colors: Sequence[Sequence[float]] = CLASS_COLORS
) -> torch.Tensor:
    """Label mask ``(H, W)`` -> colour image ``(3, H, W)`` float in ``[0, 1]``."""
    palette = torch.tensor(colors, dtype=torch.float32, device=mask.device)
    return palette[mask.long()].permute(2, 0, 1)


def overlay_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.5,
    colors: Sequence[Sequence[float]] = CLASS_COLORS,
    skip_class: int | None = BACKGROUND_CLASS,
) -> torch.Tensor:
    """Blend class colours onto an RGB image ``(3, H, W)`` in ``[0, 1]``.

    Pixels of ``skip_class`` (the background by default) keep the original image, so the
    pet and its border stand out.
    """
    color = mask_to_rgb(mask, colors).to(image.device)
    blended = (1 - alpha) * image + alpha * color
    if skip_class is None:
        return blended
    keep = (mask == skip_class).to(image.device).unsqueeze(0)
    return torch.where(keep, image, blended)


def _panels(
    image: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor | None,
    mean: Sequence[float],
    std: Sequence[float],
) -> list[torch.Tensor]:
    """``[image, true mask, (prediction,) overlay]`` as ``(3, H, W)`` tensors in ``[0, 1]``."""
    rgb = denormalize(image.detach().cpu(), mean, std)
    mask = mask.detach().cpu()
    panels = [rgb, mask_to_rgb(mask)]
    if pred is not None:
        pred = pred.detach().cpu()
        panels.append(mask_to_rgb(pred))
    panels.append(overlay_mask(rgb, mask if pred is None else pred))
    return panels


def _labels(preds: torch.Tensor | None) -> torch.Tensor | None:
    """Accept logits ``(B, C, H, W)`` or labels ``(B, H, W)``."""
    if preds is None:
        return None
    return preds.argmax(dim=1) if preds.ndim == 4 else preds


def show_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor | None = None,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    n: int = 6,
    titles: Sequence[str] | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Figure:
    """One row per sample: image | true mask | prediction | overlay of the prediction.

    Without ``preds`` the columns are image | mask | overlay of the mask (data exploration).
    ``titles`` (e.g. names or per-image IoU) are written above each row's first panel.
    """
    preds = _labels(preds)
    n = min(n, images.shape[0])
    headers = ["image", "true mask"] + (["prediction"] if preds is not None else []) + ["overlay"]
    fig, axes = plt.subplots(n, len(headers), figsize=(2.6 * len(headers), 2.6 * n), squeeze=False)
    for i in range(n):
        pred_i = None if preds is None else preds[i]
        for j, panel in enumerate(_panels(images[i], masks[i], pred_i, mean, std)):
            ax = axes[i, j]
            ax.imshow(panel.permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(headers[j], fontsize=10)
        if titles is not None:
            axes[i, 0].set_ylabel(titles[i], fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS[: len(class_names)]]
    fig.legend(handles, class_names, loc="lower center", ncol=len(class_names), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def make_tb_grid(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    n: int = 4,
) -> torch.Tensor:
    """Image grid ``(3, H', W')`` for ``SummaryWriter.add_image``.

    Each row: image | true mask | prediction | overlay (``preds`` may be logits or labels).
    """
    preds = _labels(preds)
    n = min(n, images.shape[0])
    tiles = [p for i in range(n) for p in _panels(images[i], masks[i], preds[i], mean, std)]
    return make_grid(torch.stack(tiles), nrow=4, padding=2, pad_value=1.0)


def plot_confusion_matrix(
    cm: torch.Tensor | Sequence[Sequence[int]],
    class_names: Sequence[str] = CLASS_NAMES,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> Figure:
    """Heat-map of the confusion matrix (rows = true, columns = predicted).

    With ``normalize=True`` each row is divided by its sum, so the diagonal shows the
    per-class recall and the off-diagonal cells show *what* a class is confused with.
    """
    cm = torch.as_tensor(cm, dtype=torch.float64)
    if normalize:
        rows = cm.sum(dim=1, keepdim=True)
        cm = torch.where(rows > 0, cm / rows.clamp(min=1), torch.zeros_like(cm))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm.numpy(), cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    k = len(class_names)
    ax.set_xticks(range(k), class_names)
    ax.set_yticks(range(k), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    threshold = cm.max().item() / 2
    for t in range(k):
        for p in range(k):
            value = cm[t, p].item()
            text = f"{value:.2f}" if normalize else f"{int(value)}"
            ax.text(
                p,
                t,
                text,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    fig.tight_layout()
    return fig


def plot_history(history: Mapping[str, Sequence[float]]) -> Figure:
    """Training curves from the history dict written by ``train.py``.

    Expected keys (per-epoch lists, missing ones are skipped): ``train_loss``,
    ``val_loss``, ``val_mIoU``, ``val_dice``, ``lr``. Other entries (e.g. ``best_epoch`` and
    ``run_id`` in the exported ``models/unet_pet_history.json``) are ignored.
    """
    panels = [
        ("loss", ["train_loss", "val_loss"]),
        ("validation metric", ["val_mIoU", "val_dice"]),
        ("learning rate", ["lr"]),
    ]
    plotted = {k for _, keys in panels for k in keys}
    lengths = [len(v) for k, v in history.items() if k in plotted]
    epochs = list(range(1, 1 + max(lengths, default=0)))
    panels = [(t, [k for k in keys if k in history]) for t, keys in panels]
    panels = [(t, keys) for t, keys in panels if keys]
    fig, axes = plt.subplots(
        1, max(1, len(panels)), figsize=(4.5 * max(1, len(panels)), 3.4), squeeze=False
    )
    for ax, (title, keys) in zip(axes[0], panels, strict=False):
        for key in keys:
            ax.plot(epochs[: len(history[key])], history[key], marker="o", markersize=3, label=key)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if title == "learning rate":
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig
