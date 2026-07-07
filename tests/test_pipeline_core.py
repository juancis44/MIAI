"""Tests for the generic Pipeline/PipelineStage orchestration and the
config-driven build_pipeline factory, using small no-op stages."""

from __future__ import annotations

import pytest

from miai_core.config import MIAIBaseConfig
from miai_pipeline.config import PipelineConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import UnknownStageError
from miai_pipeline.pipeline import Pipeline
from miai_pipeline.stage import PipelineStage


class _AddConfig(MIAIBaseConfig):
    key: str
    value: int


class _AddStage(PipelineStage):
    """Test-only stage: writes ``config.value`` under ``config.key``."""

    name = "add"
    config_cls = _AddConfig

    def __init__(self, config: _AddConfig) -> None:
        self.config = config

    def run(self, context: PipelineContext) -> PipelineContext:
        context.set(self.config.key, self.config.value)
        return context


class _DoubleStage(PipelineStage):
    """Test-only stage: doubles the ``total`` context key."""

    name = "double"

    def run(self, context: PipelineContext) -> PipelineContext:
        context.set("total", context.require("total") * 2)
        return context


def test_pipeline_runs_stages_in_order() -> None:
    pipeline = Pipeline(
        [
            _AddStage(_AddConfig(key="total", value=3)),
            _DoubleStage(),
            _DoubleStage(),
        ]
    )

    result = pipeline.run()

    assert result.get("total") == 12


def test_pipeline_accepts_initial_context() -> None:
    ctx = PipelineContext()
    ctx.set("total", 5)

    pipeline = Pipeline([_DoubleStage()])
    result = pipeline.run(ctx)

    assert result.get("total") == 10


def test_build_pipeline_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from miai_pipeline import stages as stages_module

    monkeypatch.setitem(stages_module.STAGE_REGISTRY, "add", _AddStage)
    monkeypatch.setitem(stages_module.STAGE_REGISTRY, "double", _DoubleStage)

    config = PipelineConfig.model_validate(
        {
            "stages": [
                {"type": "add", "params": {"key": "total", "value": 2}},
                {"type": "double", "params": {}},
            ]
        }
    )

    pipeline = Pipeline.from_config(config)
    result = pipeline.run()

    assert result.get("total") == 4


def test_build_pipeline_unknown_stage_type_raises() -> None:
    config = PipelineConfig.model_validate({"stages": [{"type": "does_not_exist", "params": {}}]})

    with pytest.raises(UnknownStageError, match="does_not_exist"):
        Pipeline.from_config(config)
