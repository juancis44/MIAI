"""Segmentation evaluation metrics: Dice and Hausdorff distance."""

from __future__ import annotations

import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric

from miai_core.config import MIAIBaseConfig


class MetricsConfig(MIAIBaseConfig):
    """Which metrics :func:`compute_case_metrics` computes.

    Attributes:
        include_dice: Whether to compute the Dice similarity coefficient.
        include_hausdorff: Whether to compute the Hausdorff distance.
        hausdorff_percentile: Percentile of the surface distance
            distribution to report (``95.0``, i.e. "HD95", is the
            conventional choice in segmentation literature -- more
            robust to single-voxel outliers than the exact maximum,
            ``100.0``).
    """

    include_dice: bool = True
    include_hausdorff: bool = True
    hausdorff_percentile: float = 95.0


def compute_case_metrics(
    prediction: torch.Tensor, ground_truth: torch.Tensor, config: MetricsConfig
) -> dict[str, float]:
    """Compute the configured metrics for a single case.

    Args:
        prediction: Binary prediction mask, shape ``(1, 1, D, H, W)``.
        ground_truth: Binary ground truth mask, same shape as
            ``prediction``.
        config: Which metrics to compute.

    Returns:
        A dict of metric name -> value (``"dice"`` and/or
        ``"hausdorff_distance"``, depending on ``config``). Both are
        ``NaN`` for a case where neither the prediction nor the ground
        truth has any foreground voxels -- that is MONAI's own
        convention for an undefined comparison, not a MIAI-specific
        one.
    """
    metrics: dict[str, float] = {}

    if config.include_dice:
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        dice_metric(y_pred=prediction, y=ground_truth)
        aggregated = dice_metric.aggregate()
        dice_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        metrics["dice"] = float(dice_tensor.item())

    if config.include_hausdorff:
        hausdorff_metric = HausdorffDistanceMetric(
            include_background=True,
            percentile=config.hausdorff_percentile,
            reduction="mean",
            get_not_nans=False,
        )
        hausdorff_metric(y_pred=prediction, y=ground_truth)
        aggregated = hausdorff_metric.aggregate()
        hausdorff_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        metrics["hausdorff_distance"] = float(hausdorff_tensor.item())

    return metrics
