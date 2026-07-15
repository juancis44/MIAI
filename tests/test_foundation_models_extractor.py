"""Unit tests for FeatureExtractor's slicing/pooling logic.

Uses a tiny fake model + processor (not a real Hugging Face download)
so these tests run fast and offline -- only the fake construction
differs from real usage, which goes through
:meth:`FeatureExtractor.from_pretrained` instead of the constructor
used here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from miai_foundation_models.exceptions import FoundationModelError
from miai_foundation_models.extractor import FeatureExtractor, FeatureExtractorConfig

_HIDDEN_SIZE = 6
_NUM_TOKENS = 4


class _FakeOutput:
    def __init__(self, last_hidden_state: torch.Tensor, pooler_output: torch.Tensor | None) -> None:
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output


class _FakeModel(torch.nn.Module):
    """A model whose per-slice embedding is a deterministic function of
    the slice's own mean pixel value, so pooling behavior can be
    checked without needing a real vision backbone."""

    def __init__(self, has_pooler: bool = True) -> None:
        super().__init__()
        self.has_pooler = has_pooler
        self.linear = torch.nn.Linear(1, _HIDDEN_SIZE)

    def forward(self, pixel_values: torch.Tensor) -> _FakeOutput:
        batch = pixel_values.shape[0]
        per_slice_scalar = pixel_values.mean(dim=(1, 2, 3), keepdim=True).reshape(batch, 1)
        token = self.linear(per_slice_scalar)
        last_hidden_state = token.unsqueeze(1).repeat(1, _NUM_TOKENS, 1)
        pooler_output = token if self.has_pooler else None
        return _FakeOutput(last_hidden_state, pooler_output)


class _FakeProcessor:
    def __call__(self, images: list[Any], return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        array = np.stack(images, axis=0).astype(np.float32)  # (B, H, W, 3)
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2)  # (B, 3, H, W)
        return {"pixel_values": tensor}


def _make_extractor(**config_kwargs: Any) -> FeatureExtractor:
    config = FeatureExtractorConfig(**config_kwargs)
    return FeatureExtractor(_FakeModel(), _FakeProcessor(), config)


def test_extract_volume_embedding_has_expected_shape() -> None:
    extractor = _make_extractor(batch_size=2)
    volume = np.random.default_rng(0).random((5, 8, 8)).astype(np.float32)

    embedding = extractor.extract_volume_embedding(volume)

    assert embedding.shape == (_HIDDEN_SIZE,)


def test_mean_slice_pooling_matches_manual_average() -> None:
    extractor = _make_extractor(slice_pooling="mean", batch_size=3)
    volume = np.random.default_rng(1).random((4, 6, 6)).astype(np.float32)

    embedding = extractor.extract_volume_embedding(volume)

    # Recompute per-slice embeddings one at a time (batch_size=1) and
    # average manually; should match the batched mean-pooled result.
    single_extractor = _make_extractor(slice_pooling="mean", batch_size=1)
    manual = torch.stack(
        [single_extractor.extract_volume_embedding(volume[i : i + 1]) for i in range(4)]
    ).mean(dim=0)

    assert torch.allclose(embedding, manual, atol=1e-5)


def test_max_slice_pooling_differs_from_mean_for_varying_slices() -> None:
    volume = np.zeros((3, 6, 6), dtype=np.float32)
    volume[1] += 1.0  # one brighter slice
    mean_extractor = _make_extractor(slice_pooling="mean")
    max_extractor = _make_extractor(slice_pooling="max")

    mean_embedding = mean_extractor.extract_volume_embedding(volume)
    max_embedding = max_extractor.extract_volume_embedding(volume)

    assert not torch.allclose(mean_embedding, max_embedding)


def test_mean_token_pooling_used_when_pooler_output_missing() -> None:
    config = FeatureExtractorConfig(token_pooling="cls")
    extractor = FeatureExtractor(_FakeModel(has_pooler=False), _FakeProcessor(), config)
    volume = np.ones((2, 4, 4), dtype=np.float32)

    embedding = extractor.extract_volume_embedding(volume)

    assert embedding.shape == (_HIDDEN_SIZE,)


def test_empty_volume_along_slice_axis_raises() -> None:
    extractor = _make_extractor()
    empty_volume = np.zeros((0, 8, 8), dtype=np.float32)

    with pytest.raises(FoundationModelError):
        extractor.extract_volume_embedding(empty_volume)


def test_unknown_slice_pooling_raises() -> None:
    extractor = _make_extractor()
    # Bypass pydantic validation to reach the runtime guard, matching
    # the pattern used for other Literal-typed config fields in this
    # codebase (e.g. NoiseScheduleConfig.schedule).
    extractor.config = extractor.config.model_copy(update={"slice_pooling": "not_a_real_strategy"})
    volume = np.ones((2, 4, 4), dtype=np.float32)

    with pytest.raises(FoundationModelError):
        extractor.extract_volume_embedding(volume)
