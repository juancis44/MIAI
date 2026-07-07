"""The stage contract every pipeline step implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from miai_core.config import MIAIBaseConfig
from miai_pipeline.context import PipelineContext


class PipelineStage(ABC):
    """Base class for a single step in a MIAI pipeline.

    Subclasses document, in their class docstring, which
    :class:`~miai_pipeline.context.PipelineContext` keys they read and
    which they write, so a pipeline's stage list can be understood by
    reading each stage in isolation.

    Attributes:
        name: A short, stable identifier for this stage, used in logs
            and in :data:`miai_pipeline.stages.STAGE_REGISTRY`.
        config_cls: The :class:`~miai_core.config.MIAIBaseConfig`
            subclass this stage is configured with, or ``None`` if the
            stage takes no configuration. Used by
            :func:`miai_pipeline.pipeline.build_pipeline` to validate a
            stage's parameters when building a :class:`Pipeline` from a
            YAML config.
    """

    name: ClassVar[str]
    config_cls: ClassVar[type[MIAIBaseConfig] | None] = None

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute this stage against the shared pipeline context.

        Args:
            context: The context produced by earlier stages (or a fresh
                one, for the first stage in a pipeline).

        Returns:
            The context, with this stage's outputs written to it.
            Implementations may mutate and return the same object, or
            return a new one — callers should always use the returned
            value.
        """
