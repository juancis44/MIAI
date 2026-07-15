"""Tests for extract_embeddings_for_paths, using a fake extractor.

Does not touch the network: FeatureExtractor.from_pretrained (the real
Hugging Face download path) is exercised only indirectly, by
higher-level code that is free to monkeypatch it out in tests -- see
test_pipeline_feature_extraction_stage.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy.typing as npt
import pytest
import torch

from conftest import make_offset_cube_volume
from miai_foundation_models.exceptions import FoundationModelError
from miai_foundation_models.run import extract_embeddings_for_paths

_EMBEDDING_DIM = 4


class _FakeExtractor:
    def extract_volume_embedding(self, volume: npt.NDArray[Any]) -> torch.Tensor:
        return torch.arange(_EMBEDDING_DIM, dtype=torch.float32)


def test_extract_embeddings_for_paths_writes_one_file_per_case(tmp_path: Path) -> None:
    image_path = make_offset_cube_volume(tmp_path / "data", name="case0", size=(4, 4, 4))

    embedding_paths = extract_embeddings_for_paths(
        _FakeExtractor(), [str(image_path)], str(tmp_path / "embeddings")
    )

    assert len(embedding_paths) == 1
    assert embedding_paths[0].exists()
    saved = torch.load(embedding_paths[0])
    assert saved.shape == (_EMBEDDING_DIM,)


def test_extract_embeddings_for_paths_empty_list_raises(tmp_path: Path) -> None:
    with pytest.raises(FoundationModelError):
        extract_embeddings_for_paths(_FakeExtractor(), [], str(tmp_path / "embeddings"))
