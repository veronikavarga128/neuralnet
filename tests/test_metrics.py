"""Tests for unet_pet.metrics on small hand-computed examples.

Reference example (6 pixels, 3 classes)::

    targets = [[0, 0, 1],      preds = [[0, 1, 1],
               [1, 2, 2]]               [1, 2, 0]]

    (true, pred) pairs: (0,0) (0,1) (1,1) (1,1) (2,2) (2,0)

    confusion matrix (rows = true, cols = pred):
        [[1, 1, 0],
         [0, 2, 0],
         [1, 0, 1]]

    class   TP  FP  FN   IoU = TP/(TP+FP+FN)   Dice = 2TP/(2TP+FP+FN)
      0      1   1   1   1/3                   2/4 = 0.5
      1      2   1   0   2/3                   4/5 = 0.8
      2      1   0   1   1/2                   2/3
    mIoU = (1/3 + 2/3 + 1/2) / 3 = 0.5;  mean Dice = (0.5 + 0.8 + 2/3) / 3;  pixel acc = 4/6
"""

import math

import pytest
import torch

from unet_pet.metrics import (
    SegmentationMeter,
    confusion_matrix,
    dice_per_class,
    iou_per_class,
    mean_dice,
    mean_iou,
    normalize_confusion,
    per_image_iou,
    pixel_accuracy,
)

TARGETS = torch.tensor([[[0, 0, 1], [1, 2, 2]]])
PREDS = torch.tensor([[[0, 1, 1], [1, 2, 0]]])
EXPECTED_CM = [[1, 1, 0], [0, 2, 0], [1, 0, 1]]


def test_confusion_matrix_hand_computed() -> None:
    cm = confusion_matrix(PREDS, TARGETS, 3)
    assert cm.tolist() == EXPECTED_CM
    assert cm.dtype == torch.int64


def test_metrics_hand_computed() -> None:
    cm = torch.tensor(EXPECTED_CM)
    assert iou_per_class(cm).tolist() == pytest.approx([1 / 3, 2 / 3, 1 / 2])
    assert dice_per_class(cm).tolist() == pytest.approx([0.5, 0.8, 2 / 3])
    assert mean_iou(cm) == pytest.approx(0.5)
    assert mean_dice(cm) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)
    assert pixel_accuracy(cm) == pytest.approx(4 / 6)


def test_confusion_matrix_accepts_logits() -> None:
    logits = torch.nn.functional.one_hot(PREDS, 3).permute(0, 3, 1, 2).float()
    assert confusion_matrix(logits, TARGETS, 3).tolist() == EXPECTED_CM


def test_perfect_prediction() -> None:
    target = torch.randint(0, 3, (2, 8, 8))
    target[0, 0, :3] = torch.tensor([0, 1, 2])  # make sure every class occurs
    cm = confusion_matrix(target, target, 3)
    assert mean_iou(cm) == 1.0 and mean_dice(cm) == 1.0 and pixel_accuracy(cm) == 1.0


def test_completely_wrong_prediction() -> None:
    target = torch.tensor([[0, 1, 2, 0]])
    cm = confusion_matrix((target + 1) % 3, target, 3)
    assert mean_iou(cm) == 0.0 and pixel_accuracy(cm) == 0.0


def test_absent_class_is_nan_and_skipped() -> None:
    target = torch.tensor([[0, 0, 1, 1]])
    cm = confusion_matrix(target, target, 3)  # class 2 never occurs and is never predicted
    ious = iou_per_class(cm)
    assert math.isnan(ious[2].item())
    assert mean_iou(cm) == 1.0  # not 2/3


def test_false_positive_on_absent_class_counts_as_zero() -> None:
    target = torch.tensor([[0, 0, 1, 1]])
    pred = torch.tensor([[0, 0, 1, 2]])  # class 2 predicted but absent -> IoU 0, not NaN
    assert iou_per_class(confusion_matrix(pred, target, 3))[2].item() == 0.0


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="Shape"):
        confusion_matrix(torch.zeros(1, 2, 2), torch.zeros(1, 3, 3), 3)
    with pytest.raises(ValueError, match="Labels"):
        confusion_matrix(torch.tensor([3]), torch.tensor([0]), 3)


def test_normalize_confusion_rows_sum_to_one() -> None:
    norm = normalize_confusion(torch.tensor(EXPECTED_CM))
    assert norm.sum(dim=1).tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert norm[0].tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert normalize_confusion(torch.zeros(3, 3, dtype=torch.long)).sum() == 0  # no NaN


def test_per_image_iou_matches_single_image_metric() -> None:
    other_target = torch.tensor([[[0, 0, 0], [1, 1, 1]]])  # no border class at all
    other_pred = torch.tensor([[[0, 0, 1], [1, 1, 1]]])
    targets = torch.cat([TARGETS, other_target])
    preds = torch.cat([PREDS, other_pred])
    ious = per_image_iou(preds, targets, 3)
    assert ious.shape == (2,)
    assert ious[0].item() == pytest.approx(0.5)
    # image 2: class 0 IoU 2/3, class 1 IoU 3/4, class 2 absent -> skipped
    assert ious[1].item() == pytest.approx((2 / 3 + 3 / 4) / 2)


def test_meter_uses_whole_split_not_batch_average() -> None:
    meter = SegmentationMeter(3)
    big_t = torch.zeros(1, 10, 10, dtype=torch.long)
    big_t[0, :5] = 1
    meter.update(big_t.clone(), big_t)  # a large, perfectly predicted image
    meter.update(PREDS, TARGETS)  # the small reference image (mIoU 0.5)
    res = meter.compute()

    expected_cm = confusion_matrix(big_t, big_t, 3) + torch.tensor(EXPECTED_CM)
    assert res["confusion_matrix"] == expected_cm.tolist()
    assert res["mIoU"] == pytest.approx(mean_iou(expected_cm))
    batch_average = (1.0 + 0.5) / 2
    assert res["mIoU"] != pytest.approx(batch_average)
    assert res["per_image_iou"] == pytest.approx([1.0, 0.5])
    assert res["pixel_acc"] == pytest.approx((100 + 4) / 106)


def test_meter_reset_and_logits() -> None:
    meter = SegmentationMeter(3)
    meter.update(torch.randn(2, 3, 4, 4), torch.randint(0, 3, (2, 4, 4)))
    meter.reset()
    assert meter.cm.sum() == 0 and meter.image_ious == []
    logits = torch.nn.functional.one_hot(PREDS, 3).permute(0, 3, 1, 2).float()
    meter.update(logits, TARGETS)
    assert meter.compute()["mIoU"] == pytest.approx(0.5)
