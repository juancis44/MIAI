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


def test_compute_case_metrics_new_metrics_off_by_default() -> None:
    mask = _mask(foreground=True)
    metrics = compute_case_metrics(mask, mask.clone(), MetricsConfig())

    assert "iou" not in metrics
    assert "sensitivity" not in metrics
    assert "specificity" not in metrics
    assert "volume_similarity" not in metrics
