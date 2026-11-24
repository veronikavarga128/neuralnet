"""Tests for unet_pet.losses (small tensors, CPU)."""

import math

import pytest
import torch
import torch.nn.functional as F

from unet_pet.config import load_config
from unet_pet.losses import CombinedLoss, DiceLoss, build_loss


def _confident_logits(labels: torch.Tensor, num_classes: int = 3, scale: float = 30.0):
    """Logits whose softmax is (almost exactly) the one-hot encoding of ``labels``."""
    return F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float() * scale


@pytest.fixture
def target() -> torch.Tensor:
    torch.manual_seed(0)
    t = torch.randint(0, 3, (2, 16, 16))
    t[0, 0, :3] = torch.tensor([0, 1, 2])
    return t


def test_perfect_prediction_gives_zero_dice_loss(target) -> None:
    assert DiceLoss()(_confident_logits(target), target).item() == pytest.approx(0.0, abs=1e-4)


def test_completely_wrong_prediction_gives_dice_loss_near_one(target) -> None:
    wrong = _confident_logits((target + 1) % 3)
    assert DiceLoss()(wrong, target).item() == pytest.approx(1.0, abs=1e-2)


def test_dice_loss_hand_computed() -> None:
    """2 pixels, 2 classes, logits 0 -> p = 0.5 everywhere; target = [0, 1].

    Per class: intersection = 0.5, sum(p) = 1, sum(onehot) = 1
    -> Dice = 2 * 0.5 / (1 + 1) = 0.5 for both classes -> loss = 1 - 0.5 = 0.5.
    """
    logits = torch.zeros(1, 2, 1, 2)
    target = torch.tensor([[[0, 1]]])
    assert DiceLoss(smooth=0.0)(logits, target).item() == pytest.approx(0.5)


def test_absent_class_is_finite_and_does_not_hurt() -> None:
    target = torch.zeros(1, 8, 8, dtype=torch.long)  # only class 0 present
    loss = DiceLoss()(_confident_logits(target), target)
    assert math.isfinite(loss.item()) and loss.item() == pytest.approx(0.0, abs=1e-4)
    # also with smooth = 0 the clamp keeps it finite
    assert math.isfinite(DiceLoss(smooth=0.0)(_confident_logits(target), target).item())


def test_dice_loss_is_in_unit_interval(target) -> None:
    loss = DiceLoss()(torch.randn(2, 3, 16, 16), target).item()
    assert 0.0 <= loss <= 1.0


def test_gradient_flows_and_is_finite(target) -> None:
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)
    CombinedLoss()(logits, target).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_combined_is_ce_plus_weighted_dice(target) -> None:
    logits = torch.randn(2, 3, 16, 16)
    ce = F.cross_entropy(logits, target)
    dice = DiceLoss()(logits, target)
    combined = CombinedLoss(dice_weight=0.5)
    assert combined(logits, target).item() == pytest.approx((ce + 0.5 * dice).item())
    assert combined.last_parts == pytest.approx({"ce": ce.item(), "dice": dice.item()})


def test_zero_dice_weight_is_plain_cross_entropy(target) -> None:
    logits = torch.randn(2, 3, 16, 16)
    assert CombinedLoss(dice_weight=0.0)(logits, target).item() == pytest.approx(
        F.cross_entropy(logits, target).item()
    )


def test_build_loss_uses_config() -> None:
    assert build_loss(load_config(overrides={"train.dice_weight": 0.25})).dice_weight == 0.25


def test_invalid_arguments_raise(target) -> None:
    with pytest.raises(ValueError):
        DiceLoss(smooth=-1)
    with pytest.raises(ValueError):
        CombinedLoss(dice_weight=-1)
    with pytest.raises(ValueError, match="Expected"):
        DiceLoss()(torch.randn(2, 3, 16, 16), target[:, :8])
