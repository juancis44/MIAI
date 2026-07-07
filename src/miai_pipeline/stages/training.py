"""Training stage interface (Phase 4 placeholder).

Concrete, working implementations (e.g. a MONAI-based training loop)
land in Phase 4, once miai-pipeline integrates with MONAI (see
docs/roadmap.md). This module defines the contract now so that
:class:`~miai_pipeline.stages.inference.InferenceStage` and
:class:`~miai_pipeline.stages.evaluation.EvaluationStage` — and anyone
writing a pipeline config today — can rely on a stable stage name and
context key names ahead of the concrete implementation landing.
"""

from __future__ import annotations

from miai_pipeline.context import PipelineContext
from miai_pipeline.stage import PipelineStage


class TrainingStage(PipelineStage):
    """Train a model on the dataset split produced by an earlier stage.

    Reads:
        ``manifest`` (``dict[str, list[str]]``): the train/val/test
        split, as produced by
        :class:`~miai_pipeline.stages.dataset.DatasetStage`.

    Writes:
        ``model_checkpoint_path`` (``str``): path to the trained
        model's checkpoint, once implemented.

    Not implemented yet: subclass this in Phase 4 to provide a concrete
    training loop (e.g. wrapping ``monai.engines`` or a plain PyTorch
    loop configured via a :class:`~miai_core.config.MIAIBaseConfig`
    subclass).
    """

    name = "training"

    def run(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError(
            "TrainingStage has no concrete implementation yet. Training "
            "utilities land in Phase 4 (MONAI integration) — see "
            "docs/roadmap.md."
        )
