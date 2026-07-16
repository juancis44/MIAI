"""Tests for plot_comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from miai_visualization.comparison import PlotComparisonConfig, plot_comparison
from miai_visualization.exceptions import VisualizationError


def test_plot_comparison_writes_file_with_diff_maps(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    images = {
        "original": rng.random((8, 8, 8)).astype(np.float32),
        "reconstructed": rng.random((8, 8, 8)).astype(np.float32),
    }

    out_path = plot_comparison(images, str(tmp_path / "comparison.png"))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_comparison_without_diff_maps_writes_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    images = {
        "a": rng.random((8, 8, 8)).astype(np.float32),
        "b": rng.random((8, 8, 8)).astype(np.float32),
        "c": rng.random((8, 8, 8)).astype(np.float32),
    }

    out_path = plot_comparison(
        images,
        str(tmp_path / "comparison.png"),
        PlotComparisonConfig(include_difference_map=False),
    )

    assert out_path.exists()


def test_plot_comparison_too_few_images_raises(tmp_path: Path) -> None:
    images = {"only_one": np.zeros((8, 8, 8), dtype=np.float32)}

    with pytest.raises(VisualizationError):
        plot_comparison(images, str(tmp_path / "comparison.png"))


def test_plot_comparison_shape_mismatch_raises(tmp_path: Path) -> None:
    images = {
        "a": np.zeros((8, 8, 8), dtype=np.float32),
        "b": np.zeros((4, 4, 4), dtype=np.float32),
    }

    with pytest.raises(VisualizationError):
        plot_comparison(images, str(tmp_path / "comparison.png"))
