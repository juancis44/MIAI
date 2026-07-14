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
