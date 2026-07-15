"""Integration test for the concrete FeatureExtractionStage.

Monkeypatches FeatureExtractor.from_pretrained so this test never
downloads a real model -- keeping the stage's context-wiring behavior
testable and CI-hermetic regardless of network access, same rationale
as using a fake model/processor in test_foundation_models_extractor.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy.typing as npt
import pytest
import torch

from conftest import make_offset_cube_volume
from miai_foundation_models.extractor import FeatureExtractor, FeatureExtractorConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.exceptions import StageError
from miai_pipeline.stages.feature_extraction import (
    FeatureExtractionStage,
    FeatureExtractionStageConfig,
)

_EMBEDDING_DIM = 4


class _FakeExtractor:
    def extract_volume_embedding(self, volume: npt.NDArray[Any]) -> torch.Tensor:
        return torch.zeros(_EMBEDDING_DIM)


def test_feature_extraction_stage_writes_embedding_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        FeatureExtractor,
        "from_pretrained",
        classmethod(lambda cls, config: _FakeExtractor()),
    )

    image_path = make_offset_cube_volume(tmp_path / "data", name="case0", size=(4, 4, 4))
    ctx = PipelineContext()
    ctx.set("preprocessed_paths", [image_path])

    stage = FeatureExtractionStage(
        FeatureExtractionStageConfig(
            output_dir=str(tmp_path / "embeddings"),
            extractor=FeatureExtractorConfig(),
        )
    )

    result = stage.run(ctx)

    embedding_paths = result.require("embedding_paths")
    assert len(embedding_paths) == 1
    assert Path(embedding_paths[0]).exists()


def test_feature_extraction_stage_empty_context_key_raises(tmp_path: Path) -> None:
    ctx = PipelineContext()
    ctx.set("preprocessed_paths", [])

    stage = FeatureExtractionStage(
        FeatureExtractionStageConfig(output_dir=str(tmp_path / "embeddings"))
    )

    with pytest.raises(StageError):
        stage.run(ctx)
