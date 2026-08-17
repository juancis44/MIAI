"""Evaluation stage: scores predictions against ground truth."""

from __future__ import annotations

from miai_core.config import MIAIBaseConfig
from miai_core.logging import get_logger
from miai_datasets.manifest import manifest_split_to_data_dicts
from miai_evaluation.evaluate import evaluate_predictions
from miai_evaluation.metrics import MetricsConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stage import PipelineStage

logger = get_logger(__name__)


class EvaluationStageConfig(MIAIBaseConfig):
    """Configuration for :class:`EvaluationStage`.

    Attributes:
        metrics: Which metrics to compute.
        report_path: Where to write the JSON evaluation report. If
            ``None``, the report is only written to the pipeline
            context, not to disk.
    """

    metrics: MetricsConfig = MetricsConfig()
    report_path: str | None = None


class EvaluationStage(PipelineStage):
    """Score predictions against ground truth and report metrics.

    Reads:
        ``prediction_paths`` (``list[Path]``): from
        :class:`~miai_pipeline.stages.inference.InferenceStage`.
        ``manifest`` (``dict[str, list]``): the ``test`` split's ground
        truth labels -- requires the manifest to have been built with a
        ``label_context_key`` (see
        :class:`~miai_pipeline.stages.dataset.DatasetStage`), so each
        entry is a ``{"image": ..., "label": ...}`` mapping, in the
        same order ``prediction_paths`` was produced in.

    Writes:
        ``metrics``
        (:class:`~miai_evaluation.evaluate.EvaluationReport`):
        ``{"per_case": [...], "mean": {...}}``, as returned by
        :func:`miai_evaluation.evaluate.evaluate_predictions`.
    """

    name = "evaluation"
    config_cls = EvaluationStageConfig

    def __init__(self, config: EvaluationStageConfig) -> None:
        """Store this stage's configuration."""
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        """Run the stage; see the class docstring for its read/write contract."""
        prediction_paths = context.require("prediction_paths")
        manifest = context.require("manifest")
        test_entries = manifest.get("test", [])

        test_dicts = manifest_split_to_data_dicts(test_entries)
        if any("label" not in d for d in test_dicts):
            raise StageError(
                "EvaluationStage requires ground truth labels in manifest['test']; "
                "build the manifest with DatasetStage's label_context_key set."
            )
        ground_truth_paths = [d["label"] for d in test_dicts]

        metrics = evaluate_predictions(
            [str(p) for p in prediction_paths],
            ground_truth_paths,
            self.config.metrics,
            self.config.report_path,
        )

        context.set("metrics", metrics)
        return context
