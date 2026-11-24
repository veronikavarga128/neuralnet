"""Segmentation losses: Dice loss and combined CrossEntropy + Dice loss.

Why CE + Dice? Cross-entropy gives smooth, well-behaved per-pixel gradients, but every
pixel counts the same, so the large background class dominates and the thin "border"
class is easy to neglect. Dice measures region overlap per class and averages the
classes with equal weight, which directly counteracts that imbalance and is closely
related to the IoU we evaluate with. The sum keeps the stability of CE and the
imbalance-robustness of Dice.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from unet_pet.config import Config


class DiceLoss(nn.Module):
    """Soft multi-class Dice loss: ``1 - mean_c Dice_c``.

    Dice is computed on softmax probabilities against one-hot targets (the hard ``argmax``
    would not be differentiable). Intersections and sums are taken over the whole batch
    per class ("batch Dice"): the border class has only a few pixels in many images, and
    a per-image Dice would be noisy for it.

    Args:
        smooth: added to numerator and denominator. Keeps the loss defined (and equal to
            0) for a class that is absent and not predicted; with batch-level sums of
            ~10^5 pixels per class its effect on present classes is negligible.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        if smooth < 0:
            raise ValueError("smooth must be >= 0")
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``logits (B, C, H, W)``, ``target (B, H, W)`` int64 labels -> scalar loss."""
        if logits.ndim != 4 or target.shape != (logits.shape[0], *logits.shape[2:]):
            raise ValueError(
                f"Expected logits (B,C,H,W) and target (B,H,W); got {tuple(logits.shape)}"
                f" and {tuple(target.shape)}"
            )
        num_classes = logits.shape[1]
        probs = logits.softmax(dim=1)
        one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).to(probs.dtype)
        dims = (0, 2, 3)  # sum over batch and pixels, keep classes
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth).clamp(min=1e-7)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """``CrossEntropy + dice_weight * Dice``; ``dice_weight = 0`` gives plain CE.

    After every call the two parts are available (detached, as floats) in
    ``last_parts = {"ce": ..., "dice": ...}`` for logging.
    """

    def __init__(self, dice_weight: float = 1.0, smooth: float = 1.0) -> None:
        super().__init__()
        if dice_weight < 0:
            raise ValueError("dice_weight must be >= 0")
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.last_parts: dict[str, float] = {}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        dice = self.dice(logits, target)
        self.last_parts = {"ce": ce.item(), "dice": dice.item()}
        return ce + self.dice_weight * dice


def build_loss(cfg: Config) -> CombinedLoss:
    """Create the training loss from the ``train`` section of the config."""
    return CombinedLoss(dice_weight=cfg.train.dice_weight)
