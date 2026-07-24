"""Segmentation evaluation metrics: overlap, boundary, and volume measures."""

from __future__ import annotations

import torch
from monai.metrics import ConfusionMatrixMetric, DiceMetric, HausdorffDistanceMetric, MeanIoU

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
        include_iou: Whether to compute the Intersection-over-Union
            (Jaccard index). Related to Dice (``IoU = Dice / (2 - Dice)``)
            but reported separately since some clinical/reporting
            conventions expect IoU rather than Dice.
        include_sensitivity: Whether to compute sensitivity (recall /
            true positive rate) -- the fraction of ground-truth
            foreground voxels the prediction correctly captured. Useful
            alongside specificity to see whether a low Dice comes from
            under- or over-segmentation.
        include_specificity: Whether to compute specificity (true
            negative rate) -- the fraction of ground-truth background
            voxels the prediction correctly left as background.
        include_volume_similarity: Whether to compute volume similarity
            (``1 - |Vp - Vg| / (Vp + Vg)``, Taha & Hanbury's definition),
            a purely count-based measure of size agreement that ignores
            spatial overlap entirely -- two masks of the same size but
            zero overlap still score close to the metric's floor for
            volume similarity, near 1.0, while scoring 0.0 on Dice/IoU.
            Complements the overlap-based metrics rather than replacing
            them.

    Every ``include_*`` flag defaults to ``False`` except the two
    original metrics (``include_dice``, ``include_hausdorff``), so
    existing configs and reports keep the same metric set unless a
    caller opts in to the new ones.
    """

    include_dice: bool = True
    include_hausdorff: bool = True
    hausdorff_percentile: float = 95.0
    include_iou: bool = False
    include_sensitivity: bool = False
    include_specificity: bool = False
    include_volume_similarity: bool = False


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

    if config.include_iou:
        iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
        iou_metric(y_pred=prediction, y=ground_truth)
        aggregated = iou_metric.aggregate()
        iou_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        metrics["iou"] = float(iou_tensor.item())

    if config.include_sensitivity:
        metrics["sensitivity"] = _confusion_matrix_metric(prediction, ground_truth, "sensitivity")

    if config.include_specificity:
        metrics["specificity"] = _confusion_matrix_metric(prediction, ground_truth, "specificity")

    if config.include_volume_similarity:
        metrics["volume_similarity"] = _volume_similarity(prediction, ground_truth)

    return metrics


def _confusion_matrix_metric(
    prediction: torch.Tensor, ground_truth: torch.Tensor, metric_name: str
) -> float:
    """Compute a single named confusion-matrix metric via MONAI.

    A thin wrapper around :class:`monai.metrics.ConfusionMatrixMetric`,
    which -- unlike :class:`~monai.metrics.DiceMetric` /
    :class:`~monai.metrics.HausdorffDistanceMetric` -- can score
    several metric names at once, so ``aggregate()`` returns a
    ``list`` (one entry per requested name) rather than a bare
    ``Tensor | tuple[Tensor, ...]``. With a single ``metric_name`` and
    ``get_not_nans=False`` that list always has exactly one entry,
    which is itself a plain ``Tensor`` (the ``tuple`` variant in its
    type signature only applies when ``get_not_nans=True``).
    """
    metric = ConfusionMatrixMetric(
        include_background=True, metric_name=metric_name, reduction="mean", get_not_nans=False
    )
    metric(y_pred=prediction, y=ground_truth)
    aggregated = metric.aggregate()[0]
    tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
    return float(tensor.item())


def _volume_similarity(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Taha & Hanbury's volume similarity: ``1 - |Vp - Vg| / (Vp + Vg)``.

    Purely count-based (voxel counts, not spatial overlap) -- computed
    directly rather than via a MONAI metric, since MONAI does not
    expose this one. Returns ``1.0`` when both masks are entirely
    background (no volume to disagree on), matching this module's
    existing convention of returning a defined value rather than NaN
    wherever a natural default exists (MONAI itself returns NaN for
    Dice/Hausdorff in that case, since overlap of two empty sets truly
    is undefined -- volume similarity has no such ambiguity).
    """
    prediction_volume = float(prediction.sum().item())
    ground_truth_volume = float(ground_truth.sum().item())
    denominator = prediction_volume + ground_truth_volume
    if denominator == 0.0:
        return 1.0
    return 1.0 - abs(prediction_volume - ground_truth_volume) / denominator
