"""Inference stage interface (Phase 4 placeholder).

See :mod:`miai_pipeline.stages.training` for why this is an interface
only for now.
"""

from __future__ import annotations

from miai_pipeline.context import PipelineContext
from miai_pipeline.stage import PipelineStage


class InferenceStage(PipelineStage):
    """Run a trained model over the test split to produce predictions.

    Reads:
        ``model_checkpoint_path`` (``str``): the trained model, as
        produced by
        :class:`~miai_pipeline.stages.training.TrainingStage`.
        ``manifest`` (``dict[str, list[str]]``): provides the ``test``
        cases to run inference on.

    Writes:
        ``prediction_paths`` (``list[Path]``): one prediction volume
        per test case, once implemented.

    Not implemented yet: see docs/roadmap.md Phase 4.
    """

    name = "inference"

    def run(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError(
            "InferenceStage has no concrete implementation yet. Inference "
            "utilities land in Phase 4 (MONAI integration) — see "
            "docs/roadmap.md."
        )
