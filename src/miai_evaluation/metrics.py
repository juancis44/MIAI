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

    Attributes:
        num_classes: Number of segmentation classes, including
            background. ``1`` (the default) is the original binary
            path -- every metric above is computed directly on
            ``prediction``/``ground_truth`` as 0/1 masks, including the
            background voxels (``include_background=True``), unchanged
            from this module's original behavior. Any value ``> 1``
            switches to a multi-class path: ``prediction`` and
            ``ground_truth`` are each read as a single-channel integer
            class-id mask (values in ``[0, num_classes)``, exactly what
            a multi-class :func:`~miai_segmentation.two_d.infer.
            run_case_inference` prediction or an ACDC-style ground
            truth label file already is -- no separate one-hot
            preprocessing needed by the caller), one-hot encoded
            internally, and every metric above is computed
            **excluding** the background channel
            (``include_background=False``) -- so e.g. mean Dice
            reflects agreement on the actual foreground structures, not
            on background voxels that make up most of the image and
            would otherwise dominate the average. ``include_dice``
            additionally reports one ``dice_class_{c}`` entry per
            foreground class ``c`` (``1`` through ``num_classes - 1``),
            since a single macro-averaged Dice hides which structure a
            model struggles with -- for ACDC's convention (background,
            right ventricle, myocardium, left ventricle), that is
            ``dice_class_1`` (RV), ``dice_class_2`` (myocardium), and
            ``dice_class_3`` (LV). This module stays dataset-agnostic
            on purpose -- callers that know their own class semantics
            (like ``examples/validate_acdc.py``) are expected to
            relabel these for human-readable reporting, not this
            module. Every other opted-in metric gets the same
            per-class treatment in multi-class mode: ``include_
            hausdorff`` additionally reports ``hausdorff_distance_
            class_{c}``, ``include_iou`` reports ``iou_class_{c}``,
            ``include_sensitivity``/``include_specificity`` report
            ``sensitivity_class_{c}``/``specificity_class_{c}``, and
            ``include_volume_similarity`` reports ``volume_
            similarity_class_{c}`` -- each computed the same way as
            ``dice_class_{c}``: on that class's one-hot channel alone
            (prediction and ground truth both restricted to "is this
            voxel class ``c``"), not derived from the macro-averaged
            value.
    """

    include_dice: bool = True
    include_hausdorff: bool = True
    hausdorff_percentile: float = 95.0
    include_iou: bool = False
    include_sensitivity: bool = False
    include_specificity: bool = False
    include_volume_similarity: bool = False
    num_classes: int = 1


def compute_case_metrics(
    prediction: torch.Tensor, ground_truth: torch.Tensor, config: MetricsConfig
) -> dict[str, float]:
    """Compute the configured metrics for a single case.

    Args:
        prediction: Prediction mask, shape ``(1, 1, D, H, W)``. Binary
            0/1 when ``config.num_classes == 1``; an integer class-id
            mask with values in ``[0, config.num_classes)`` when
            ``config.num_classes > 1`` -- see :class:`MetricsConfig`.
        ground_truth: Ground truth mask, same shape and convention as
            ``prediction``.
        config: Which metrics to compute, and (via ``num_classes``)
            whether ``prediction``/``ground_truth`` are read as binary
            or multi-class.

    Returns:
        A dict of metric name -> value (``"dice"``, ``"hausdorff_
        distance"``, and so on, depending on ``config`` -- plus one
        ``{metric}_class_{c}`` entry per foreground class per opted-in
        metric when ``config.num_classes > 1``, e.g. ``dice_class_1``,
        ``hausdorff_distance_class_1``, ``iou_class_1``, ``sensitivity_
        class_1``, ``specificity_class_1``, ``volume_similarity_
        class_1``). Every metric is ``NaN`` for a case/class where
        neither the prediction nor the ground truth has any matching
        voxels -- that is MONAI's own convention for an undefined
        comparison, not a MIAI-specific one.
    """
    multiclass = config.num_classes > 1
    if multiclass:
        pred_onehot = _to_one_hot(prediction, config.num_classes)
        gt_onehot = _to_one_hot(ground_truth, config.num_classes)
        metric_pred, metric_gt = pred_onehot, gt_onehot
        volume_pred, volume_gt = (prediction > 0).float(), (ground_truth > 0).float()
    else:
        metric_pred, metric_gt = prediction, ground_truth
        volume_pred, volume_gt = prediction, ground_truth
    include_background = not multiclass

    metrics: dict[str, float] = {}

    if config.include_dice:
        dice_metric = DiceMetric(
            include_background=include_background, reduction="mean", get_not_nans=False
        )
        dice_metric(y_pred=metric_pred, y=metric_gt)
        aggregated = dice_metric.aggregate()
        dice_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        metrics["dice"] = float(dice_tensor.item())

        if multiclass:
            for class_id in range(1, config.num_classes):
                metrics[f"dice_class_{class_id}"] = _binary_dice(
                    pred_onehot[:, class_id : class_id + 1],
                    gt_onehot[:, class_id : class_id + 1],
                )

    if config.include_hausdorff:
        hausdorff_metric = HausdorffDistanceMetric(
            include_background=include_background,
            percentile=config.hausdorff_percentile,
            reduction="mean",
            get_not_nans=False,
        )
        hausdorff_metric(y_pred=metric_pred, y=metric_gt)
        aggregated = hausdorff_metric.aggregate()
        hausdorff_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        metrics["hausdorff_distance"] = float(hausdorff_tensor.item())

        if multiclass:
            for class_id in range(1, config.num_classes):
                metrics[f"hausdorff_distance_class_{class_id}"] = _binary_hausdorff(
                    pred_onehot[:, class_id : class_id + 1],
                    gt_onehot[:, class_id : class_id + 1],
                    config.hausdorff_percentile,
                )

    if config.include_iou:
        iou_metric = MeanIoU(
            include_background=include_background, reduction="mean", get_not_nans=False
        )
        iou_metric(y_pred=metric_pred, y=metric_gt)
        aggregated = iou_metric.aggregate()
        iou_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
        metrics["iou"] = float(iou_tensor.item())

        if multiclass:
            for class_id in range(1, config.num_classes):
                metrics[f"iou_class_{class_id}"] = _binary_iou(
                    pred_onehot[:, class_id : class_id + 1],
                    gt_onehot[:, class_id : class_id + 1],
                )

    if config.include_sensitivity:
        metrics["sensitivity"] = _confusion_matrix_metric(
            metric_pred, metric_gt, "sensitivity", include_background
        )

        if multiclass:
            for class_id in range(1, config.num_classes):
                metrics[f"sensitivity_class_{class_id}"] = _confusion_matrix_metric(
                    pred_onehot[:, class_id : class_id + 1],
                    gt_onehot[:, class_id : class_id + 1],
                    "sensitivity",
                    include_background=True,
                )

    if config.include_specificity:
        metrics["specificity"] = _confusion_matrix_metric(
            metric_pred, metric_gt, "specificity", include_background
        )

        if multiclass:
            for class_id in range(1, config.num_classes):
                metrics[f"specificity_class_{class_id}"] = _confusion_matrix_metric(
                    pred_onehot[:, class_id : class_id + 1],
                    gt_onehot[:, class_id : class_id + 1],
                    "specificity",
                    include_background=True,
                )

    if config.include_volume_similarity:
        metrics["volume_similarity"] = _volume_similarity(volume_pred, volume_gt)

        if multiclass:
            for class_id in range(1, config.num_classes):
                metrics[f"volume_similarity_class_{class_id}"] = _volume_similarity(
                    pred_onehot[:, class_id : class_id + 1],
                    gt_onehot[:, class_id : class_id + 1],
                )

    return metrics


def _to_one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot encode an integer class-id mask along a new channel dim.

    ``mask`` is ``(1, 1, ...)`` with integer-valued entries in
    ``[0, num_classes)`` -- exactly what a multi-class
    :func:`~miai_segmentation.two_d.infer.run_case_inference`
    prediction, or an ACDC-style ground truth label file loaded
    straight off disk, already is. Returns ``(1, num_classes, ...)``,
    channel ``c`` a 1/0 indicator of "this voxel is class ``c``" --
    the shape every MONAI metric in this module expects for its own
    ``include_background`` handling to mean what it says (channel 0 is
    background, by convention).
    """
    labels = mask.long().squeeze(1)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes)
    return torch.movedim(one_hot, -1, 1).float()


def _binary_dice(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Dice of a single one-hot channel pair -- the per-class breakdown's building block."""
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_metric(y_pred=prediction, y=ground_truth)
    aggregated = dice_metric.aggregate()
    dice_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
    return float(dice_tensor.item())


def _binary_hausdorff(
    prediction: torch.Tensor, ground_truth: torch.Tensor, percentile: float
) -> float:
    """Hausdorff distance of a single one-hot channel pair -- ``_binary_dice``'s counterpart."""
    hausdorff_metric = HausdorffDistanceMetric(
        include_background=True, percentile=percentile, reduction="mean", get_not_nans=False
    )
    hausdorff_metric(y_pred=prediction, y=ground_truth)
    aggregated = hausdorff_metric.aggregate()
    hausdorff_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
    return float(hausdorff_tensor.item())


def _binary_iou(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Compute IoU of a single one-hot channel pair -- ``_binary_dice``'s counterpart."""
    iou_metric = MeanIoU(include_background=True, reduction="mean", get_not_nans=False)
    iou_metric(y_pred=prediction, y=ground_truth)
    aggregated = iou_metric.aggregate()
    iou_tensor = aggregated[0] if isinstance(aggregated, tuple) else aggregated
    return float(iou_tensor.item())


def _confusion_matrix_metric(
    prediction: torch.Tensor,
    ground_truth: torch.Tensor,
    metric_name: str,
    include_background: bool = True,
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
        include_background=include_background,
        metric_name=metric_name,
        reduction="mean",
        get_not_nans=False,
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
