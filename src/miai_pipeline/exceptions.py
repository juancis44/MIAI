"""Exceptions specific to pipeline orchestration."""

from __future__ import annotations

from miai_core.exceptions import MIAIError


class PipelineError(MIAIError):
    """Base class for pipeline orchestration failures."""


class StageError(PipelineError):
    """Raised when a stage fails while processing the pipeline context."""


class UnknownStageError(PipelineError):
    """Raised when a pipeline config references an unregistered stage type.

    See :data:`miai_pipeline.stages.STAGE_REGISTRY` for valid values.
    """
