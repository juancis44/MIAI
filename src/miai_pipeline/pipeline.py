"""The pipeline runner and the factory that builds one from a config."""

from __future__ import annotations

from collections.abc import Sequence

from miai_core.logging import get_logger
from miai_pipeline.config import PipelineConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import UnknownStageError
from miai_pipeline.stage import PipelineStage

logger = get_logger(__name__)


class Pipeline:
    """Runs an ordered sequence of :class:`~miai_pipeline.stage.PipelineStage`.

    A ``Pipeline`` is deliberately dumb: it just calls each stage's
    ``run`` in order, passing the context along, and logs which stage is
    executing. All actual behavior — and all configuration — lives in
    the stages themselves, which is what keeps a pipeline reproducible
    from its config file alone.
    """

    def __init__(self, stages: Sequence[PipelineStage]) -> None:
        self.stages = list(stages)

    def run(self, context: PipelineContext | None = None) -> PipelineContext:
        """Run every stage in order against a shared context.

        Args:
            context: An initial context (e.g. with ``dicom_dir`` already
                set). If ``None``, an empty context is created.

        Returns:
            The final context, after every stage has run.
        """
        ctx = context if context is not None else PipelineContext()
        for stage in self.stages:
            logger.info("Running stage: %s", stage.name)
            ctx = stage.run(ctx)
        return ctx

    @classmethod
    def from_config(cls, config: PipelineConfig) -> Pipeline:
        """Build a :class:`Pipeline` from a validated
        :class:`~miai_pipeline.config.PipelineConfig`.

        Args:
            config: The pipeline configuration, typically loaded via
                ``PipelineConfig.from_yaml(...)``.

        Returns:
            A :class:`Pipeline` with one instantiated stage per entry in
            ``config.stages``.

        Raises:
            UnknownStageError: If a stage's ``type`` is not registered
                in :data:`miai_pipeline.stages.STAGE_REGISTRY`.
        """
        from miai_pipeline.stages import STAGE_REGISTRY

        stages: list[PipelineStage] = []
        for stage_config in config.stages:
            stage_cls = STAGE_REGISTRY.get(stage_config.type)
            if stage_cls is None:
                available = ", ".join(sorted(STAGE_REGISTRY))
                raise UnknownStageError(
                    f"Unknown stage type '{stage_config.type}'. "
                    f"Available stage types: {available}."
                )

            # Each concrete stage's __init__ signature differs (some take a
            # config, some take none), so this dynamic construction from the
            # registry can't be statically typed by mypy.
            if stage_cls.config_cls is not None:
                stage_params = stage_cls.config_cls.model_validate(stage_config.params)
                stages.append(stage_cls(stage_params))  # type: ignore[call-arg]
            else:
                stages.append(stage_cls())

        return cls(stages)
