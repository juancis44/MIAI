"""Tests for plot_embedding_projection."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from miai_visualization.embeddings import PlotEmbeddingProjectionConfig, plot_embedding_projection
from miai_visualization.exceptions import VisualizationError


def test_plot_embedding_projection_writes_file(tmp_path: Path) -> None:
    torch.manual_seed(0)
    embeddings = torch.rand(10, 6)

    out_path = plot_embedding_projection(embeddings, str(tmp_path / "projection.png"))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_embedding_projection_with_labels_writes_file(tmp_path: Path) -> None:
    torch.manual_seed(0)
    embeddings = torch.rand(6, 4)
    labels = ["a", "a", "b", "b", "c", "c"]

    out_path = plot_embedding_projection(
        embeddings,
        str(tmp_path / "projection.png"),
        PlotEmbeddingProjectionConfig(title="test"),
        labels=labels,
    )

    assert out_path.exists()


def test_plot_embedding_projection_too_few_rows_raises(tmp_path: Path) -> None:
    embeddings = torch.rand(1, 4)

    with pytest.raises(VisualizationError):
        plot_embedding_projection(embeddings, str(tmp_path / "projection.png"))


def test_plot_embedding_projection_label_length_mismatch_raises(tmp_path: Path) -> None:
    embeddings = torch.rand(5, 4)

    with pytest.raises(VisualizationError):
        plot_embedding_projection(embeddings, str(tmp_path / "projection.png"), labels=["a", "b"])
