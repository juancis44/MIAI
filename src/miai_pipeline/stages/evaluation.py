"""Evaluation stage interface (Phase 4 placeholder).

See :mod:`miai_pipeline.stages.training` for why this is an interface
only for now.
"""

from __future__ import annotations

from miai_pipeline.context import PipelineContext
from miai_pipeline.stage import PipelineStage


class EvaluationStage(PipelineStage):
    """Score predictions against ground truth and report metrics.

    Reads:
        ``prediction_paths`` (``list[Path]``): from
        :class:`~miai_pipeline.stages.inference.InferenceStage`.
        ``manifest`` (``dict[str, list[str]]``): provides the ground
        truth ``test`` cases to compare predictions against.

    Writes:
        ``metrics`` (``dict[str, float]``): summary metrics, once
        implemented.

    Not implemented yet: metric implementations (Dice, Hausdorff
    distance, etc.) land alongside ``miai-evaluation`` in a later
    phase — see docs/roadmap.md.
    """

    name = "evaluation"

    def run(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError(
            "EvaluationStage has no concrete implementation yet. Metric "
            "implementations land alongside miai-evaluation in a later "
            "phase — see docs/roadmap.md."
        )
