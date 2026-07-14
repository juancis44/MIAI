"""Evaluate segmentation predictions against ground truth on disk."""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk
import torch

from miai_core.io import write_json
from miai_core.logging import get_logger
from miai_evaluation.exceptions import EvaluationError
from miai_evaluation.metrics import MetricsConfig, compute_case_metrics

logger = get_logger(__name__)


def _load_mask(path: str) -> torch.Tensor:
    array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return torch.as_tensor(array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def evaluate_predictions(
    prediction_paths: list[str],
    ground_truth_paths: list[str],
    config: MetricsConfig,
    output_path: str | None = None,
) -> dict[str, object]:
    """Score predictions against ground truth and aggregate metrics.

    Reads each prediction/ground-truth pair from disk via SimpleITK
    (consistent with the rest of MIAI's image I/O -- see
    :class:`~miai_transforms.sitk_transforms.LoadImageSitkd` and
    :func:`miai_segmentation.infer.run_inference`), scores it with
    :func:`miai_evaluation.metrics.compute_case_metrics`, and averages
    across cases.

    Args:
        prediction_paths: One prediction NIfTI per case (e.g. from
            :func:`miai_segmentation.infer.run_inference`).
        ground_truth_paths: One ground truth label NIfTI per case, in
            the same order as ``prediction_paths``.
        config: Which metrics to compute.
        output_path: If given, the returned report is also written
            here as JSON.

    Returns:
        A dict with ``"per_case"`` (a list of per-case metric dicts,
        each including a ``"case"`` key naming the prediction file) and
        ``"mean"`` (metric name -> average across cases).

    Raises:
        EvaluationError: If ``prediction_paths`` and
            ``ground_truth_paths`` have different lengths, or either is
            empty.
    """
    if len(prediction_paths) != len(ground_truth_paths):
        raise EvaluationError(
            f"prediction_paths has {len(prediction_paths)} entries but "
            f"ground_truth_paths has {len(ground_truth_paths)}; they must be aligned "
            "one ground truth per prediction."
        )
    if not prediction_paths:
        raise EvaluationError("prediction_paths is empty; nothing to evaluate.")

    per_case: list[dict[str, object]] = []
    all_metrics: list[dict[str, float]] = []
    for pred_path, gt_path in zip(prediction_paths, ground_truth_paths, strict=True):
        prediction = _load_mask(pred_path)
        ground_truth = _load_mask(gt_path)
        case_metrics = compute_case_metrics(prediction, ground_truth, config)
        all_metrics.append(case_metrics)
        per_case.append({"case": Path(pred_path).name, **case_metrics})
        logger.info("Metrics for %s: %s", pred_path, case_metrics)

    metric_names = list(all_metrics[0])
    mean_metrics = {
        name: sum(m[name] for m in all_metrics) / len(all_metrics) for name in metric_names
    }

    report: dict[str, object] = {"per_case": per_case, "mean": mean_metrics}

    if output_path is not None:
        write_json(report, output_path)

    return report
