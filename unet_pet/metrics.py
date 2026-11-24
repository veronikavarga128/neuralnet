"""Confusion-matrix based segmentation metrics (IoU, mIoU, Dice, pixel accuracy).

All metrics are derived from one confusion matrix accumulated over the *whole* split
(rows = true class, columns = predicted class). Averaging per-batch scores instead would
weight small batches and images unfairly and give a different, biased number.

mIoU is the main metric: pixel accuracy is dominated by the large background class and
looks good even for a useless model (see the "always background" baseline).
"""

from __future__ import annotations

from typing import Any

import torch


def _to_labels(preds: torch.Tensor) -> torch.Tensor:
    """Accept logits ``(B, C, H, W)`` or labels ``(B, H, W)``; return labels."""
    return preds.argmax(dim=1) if preds.ndim == 4 else preds


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``(C, C)`` int64 confusion matrix; entry ``[t, p]`` counts pixels of class t predicted p.

    Args:
        preds: predicted labels ``(...)`` or logits ``(B, C, H, W)``.
        targets: true labels, same shape as the predicted labels.
    """
    preds = _to_labels(preds)
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: preds {tuple(preds.shape)}, targets {tuple(targets.shape)}"
        )
    t, p = targets.reshape(-1).long(), preds.reshape(-1).long()
    if t.numel() and (
        t.min() < 0 or t.max() >= num_classes or p.min() < 0 or p.max() >= num_classes
    ):
        raise ValueError(f"Labels must be in [0, {num_classes - 1}]")
    return torch.bincount(t * num_classes + p, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Elementwise ratio; 0/0 becomes NaN (the class is absent and was never predicted)."""
    num, den = numerator.double(), denominator.double()
    return torch.where(den > 0, num / den.clamp(min=1), torch.full_like(num, float("nan")))


def _tp_fp_fn(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tp = cm.diagonal()
    return tp, cm.sum(dim=0) - tp, cm.sum(dim=1) - tp  # FP: column rest, FN: row rest


def iou_per_class(cm: torch.Tensor) -> torch.Tensor:
    """IoU = TP / (TP + FP + FN) per class (NaN if the class never occurs nor is predicted)."""
    tp, fp, fn = _tp_fp_fn(cm)
    return _ratio(tp, tp + fp + fn)


def dice_per_class(cm: torch.Tensor) -> torch.Tensor:
    """Dice (= F1) = 2TP / (2TP + FP + FN) per class (NaN if undefined)."""
    tp, fp, fn = _tp_fp_fn(cm)
    return _ratio(2 * tp, 2 * tp + fp + fn)


def mean_iou(cm: torch.Tensor) -> float:
    """Mean IoU over the classes where it is defined."""
    return float(torch.nanmean(iou_per_class(cm)))


def mean_dice(cm: torch.Tensor) -> float:
    """Mean (macro) Dice over the classes where it is defined."""
    return float(torch.nanmean(dice_per_class(cm)))


def pixel_accuracy(cm: torch.Tensor) -> float:
    """Fraction of correctly classified pixels (NaN for an empty matrix)."""
    total = cm.sum()
    return float(cm.diagonal().sum() / total) if total > 0 else float("nan")


def normalize_confusion(cm: torch.Tensor) -> torch.Tensor:
    """Row-normalise: entry ``[t, p]`` = share of class-t pixels predicted as p (recall view)."""
    rows = cm.sum(dim=1, keepdim=True).double()
    return torch.where(rows > 0, cm.double() / rows.clamp(min=1), torch.zeros_like(rows))


def per_image_iou(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """mIoU of each image in a batch, ``(B,)`` float64.

    Classes absent from both the image and its prediction are skipped (NaN-aware mean),
    otherwise every image without a border pixel would be punished or rewarded arbitrarily.
    """
    preds = _to_labels(preds)
    b = targets.shape[0]
    c2 = num_classes**2
    idx = targets.reshape(b, -1).long() * num_classes + preds.reshape(b, -1).long()
    idx = idx + torch.arange(b, device=idx.device).unsqueeze(1) * c2  # separate bins per image
    cms = torch.bincount(idx.reshape(-1), minlength=b * c2).reshape(b, num_classes, num_classes)
    return torch.stack([torch.nanmean(iou_per_class(cm)) for cm in cms])


class SegmentationMeter:
    """Accumulate a confusion matrix (and per-image mIoU) over many batches.

    Usage::

        meter = SegmentationMeter(num_classes=3)
        for images, masks in loader:
            meter.update(model(images), masks)
        results = meter.compute()
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.cm = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)
        self.image_ious: list[float] = []

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Add a batch: ``preds`` as logits ``(B, C, H, W)`` or labels ``(B, H, W)``."""
        labels = _to_labels(preds)
        self.cm += confusion_matrix(labels, targets, self.num_classes).cpu()
        self.image_ious += per_image_iou(labels, targets, self.num_classes).cpu().tolist()

    def compute(self) -> dict[str, Any]:
        """Metrics over everything seen so far, as plain (JSON-serialisable) Python values."""
        return {
            "mIoU": mean_iou(self.cm),
            "dice": mean_dice(self.cm),
            "pixel_acc": pixel_accuracy(self.cm),
            "iou_per_class": iou_per_class(self.cm).tolist(),
            "dice_per_class": dice_per_class(self.cm).tolist(),
            "confusion_matrix": self.cm.tolist(),
            "per_image_iou": list(self.image_ious),
        }
