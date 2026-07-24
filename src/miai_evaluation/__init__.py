"""MIAI Evaluation: scores segmentation predictions against ground truth.

Provides Dice and Hausdorff-distance metrics
(:mod:`miai_evaluation.metrics`) and a file-based evaluation runner
(:func:`miai_evaluation.evaluate.evaluate_predictions`) that reads
prediction/ground-truth NIfTI pairs from disk via SimpleITK --
consistent with the rest of MIAI's image I/O (see
:class:`~miai_transforms.sitk_transforms.LoadImageSitkd`) -- and
aggregates per-case metrics into a summary report. Used by
:class:`~miai_pipeline.stages.evaluation.EvaluationStage` to implement
the final step of the clinical workflow.
"""

from miai_evaluation.evaluate import EvaluationReport, evaluate_predictions
from miai_evaluation.exceptions import EvaluationError
from miai_evaluation.metrics import MetricsConfig, compute_case_metrics

__version__ = "0.1.0"

__all__ = [
    "evaluate_predictions",
    "EvaluationReport",
    "EvaluationError",
    "MetricsConfig",
    "compute_case_metrics",
    "__version__",
]
