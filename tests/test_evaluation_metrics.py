"""Tests for miai_evaluation.metrics."""

import pytest
import torch

from miai_evaluation.metrics import MetricsConfig, compute_case_metrics


def _mask(*, foreground: bool) -> torch.Tensor:
    mask = torch.zeros(1, 1, 8, 8, 8)
    if foreground:
        mask[:, :, 2:6, 2:6, 2:6] = 1.0
    return mask


def test_compute_case_metrics_perfect_match() -> None:
    mask = _mask(foreground=True)
    metrics = compute_case_metrics(mask, mask.clone(), MetricsConfig())

    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["hausdorff_distance"] == pytest.approx(0.0)


def test_compute_case_metrics_no_overlap() -> None:
    prediction = torch.zeros(1, 1, 8, 8, 8)
    prediction[:, :, 0:2, 0:2, 0:2] = 1.0
    ground_truth = torch.zeros(1, 1, 8, 8, 8)
    ground_truth[:, :, 6:8, 6:8, 6:8] = 1.0

    metrics = compute_case_metrics(prediction, ground_truth, MetricsConfig())

    assert metrics["dice"] == pytest.approx(0.0)
    assert metrics["hausdorff_distance"] > 0.0


def test_compute_case_metrics_respects_config_toggles() -> None:
    mask = _mask(foreground=True)

    dice_only = compute_case_metrics(
        mask, mask.clone(), MetricsConfig(include_dice=True, include_hausdorff=False)
    )
    assert set(dice_only) == {"dice"}

    hausdorff_only = compute_case_metrics(
        mask, mask.clone(), MetricsConfig(include_dice=False, include_hausdorff=True)
    )
    assert set(hausdorff_only) == {"hausdorff_distance"}


def _all_new_metrics_config() -> MetricsConfig:
    return MetricsConfig(
        include_dice=False,
        include_hausdorff=False,
        include_iou=True,
        include_sensitivity=True,
        include_specificity=True,
        include_volume_similarity=True,
    )


def test_compute_case_metrics_new_metrics_perfect_match() -> None:
    mask = _mask(foreground=True)
    metrics = compute_case_metrics(mask, mask.clone(), _all_new_metrics_config())

    assert set(metrics) == {"iou", "sensitivity", "specificity", "volume_similarity"}
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)
    assert metrics["volume_similarity"] == pytest.approx(1.0)


def test_compute_case_metrics_new_metrics_no_overlap() -> None:
    prediction = torch.zeros(1, 1, 8, 8, 8)
    prediction[:, :, 0:2, 0:2, 0:2] = 1.0
    ground_truth = torch.zeros(1, 1, 8, 8, 8)
    ground_truth[:, :, 6:8, 6:8, 6:8] = 1.0

    metrics = compute_case_metrics(prediction, ground_truth, _all_new_metrics_config())

    assert metrics["iou"] == pytest.approx(0.0)
    assert metrics["sensitivity"] == pytest.approx(0.0)
    # Same volume (8 voxels each) and disjoint location: overlap-based metrics
    # are 0, but volume similarity only compares voxel *counts*, so it is
    # still 1.0 -- the whole point of reporting it alongside Dice/IoU.
    assert metrics["volume_similarity"] == pytest.approx(1.0)


def test_volume_similarity_penalizes_size_mismatch_not_just_overlap() -> None:
    prediction = torch.zeros(1, 1, 8, 8, 8)
    prediction[:, :, 0:2, 0:2, 0:2] = 1.0  # 8 voxels
    ground_truth = torch.zeros(1, 1, 8, 8, 8)
    ground_truth[:, :, 6:8, 6:8, 7:8] = 1.0  # 4 voxels

    metrics = compute_case_metrics(
        prediction, ground_truth, MetricsConfig(include_dice=False, include_volume_similarity=True)
    )

    assert metrics["volume_similarity"] == pytest.approx(1.0 - abs(8 - 4) / (8 + 4))


def test_volume_similarity_both_empty_returns_one() -> None:
    # Neither mask has any foreground voxels -- there's no volume to
    # disagree on, so this hits _volume_similarity's denominator==0.0
    # branch and returns the defined default (1.0) instead of dividing
    # by zero.
    prediction = torch.zeros(1, 1, 8, 8, 8)
    ground_truth = torch.zeros(1, 1, 8, 8, 8)

    metrics = compute_case_metrics(
        prediction, ground_truth, MetricsConfig(include_dice=False, include_volume_similarity=True)
    )

    assert metrics["volume_similarity"] == pytest.approx(1.0)


def test_compute_case_metrics_new_metrics_off_by_default() -> None:
    mask = _mask(foreground=True)
    metrics = compute_case_metrics(mask, mask.clone(), MetricsConfig())

    assert "iou" not in metrics
    assert "sensitivity" not in metrics
    assert "specificity" not in metrics
    assert "volume_similarity" not in metrics


def _multiclass_ground_truth() -> torch.Tensor:
    """Three disjoint foreground classes (1, 2, 3) plus background (0),
    the same integer-class-id convention ACDC's own ground truth uses
    (background, RV, myocardium, LV)."""
    mask = torch.zeros(1, 1, 8, 8, 8)
    mask[:, :, 0:2, 0:2, 0:2] = 1.0
    mask[:, :, 3:5, 3:5, 3:5] = 2.0
    mask[:, :, 6:8, 6:8, 6:8] = 3.0
    return mask


def test_compute_case_metrics_multiclass_perfect_match() -> None:
    ground_truth = _multiclass_ground_truth()
    metrics = compute_case_metrics(
        ground_truth,
        ground_truth.clone(),
        MetricsConfig(include_dice=True, include_hausdorff=False, num_classes=4),
    )

    assert set(metrics) == {"dice", "dice_class_1", "dice_class_2", "dice_class_3"}
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["dice_class_1"] == pytest.approx(1.0)
    assert metrics["dice_class_2"] == pytest.approx(1.0)
    assert metrics["dice_class_3"] == pytest.approx(1.0)


def test_compute_case_metrics_multiclass_reports_per_class_breakdown() -> None:
    """One class entirely missed -- the per-class Dice should isolate
    that failure instead of it being averaged away, and the overall
    ``dice`` should be the mean across foreground classes only (never
    touching the -- correctly predicted, and much larger -- background
    class, which excluding it from ``include_background`` guards
    against)."""
    ground_truth = _multiclass_ground_truth()
    prediction = ground_truth.clone()
    prediction[:, :, 0:2, 0:2, 0:2] = 0.0  # class 1 predicted as background instead

    metrics = compute_case_metrics(
        prediction,
        ground_truth,
        MetricsConfig(include_dice=True, include_hausdorff=False, num_classes=4),
    )

    assert metrics["dice_class_1"] == pytest.approx(0.0)
    assert metrics["dice_class_2"] == pytest.approx(1.0)
    assert metrics["dice_class_3"] == pytest.approx(1.0)
    assert metrics["dice"] == pytest.approx(2.0 / 3.0)


def test_compute_case_metrics_multiclass_no_per_class_keys_when_dice_disabled() -> None:
    ground_truth = _multiclass_ground_truth()
    metrics = compute_case_metrics(
        ground_truth,
        ground_truth.clone(),
        MetricsConfig(include_dice=False, include_hausdorff=False, include_iou=True, num_classes=4),
    )

    assert "dice" not in metrics
    assert not any(key.startswith("dice_class_") for key in metrics)
    assert metrics["iou"] == pytest.approx(1.0)


def test_compute_case_metrics_multiclass_volume_similarity_ignores_class_identity() -> None:
    """Multi-class volume similarity compares total foreground volume
    (any nonzero class), not per-class counts -- swapping which class a
    region belongs to, without changing its size, should not move it."""
    ground_truth = _multiclass_ground_truth()
    prediction = ground_truth.clone()
    prediction[prediction == 1.0] = 2.0  # relabel class 1's voxels as class 2

    metrics = compute_case_metrics(
        prediction,
        ground_truth,
        MetricsConfig(include_dice=False, include_volume_similarity=True, num_classes=4),
    )

    assert metrics["volume_similarity"] == pytest.approx(1.0)
