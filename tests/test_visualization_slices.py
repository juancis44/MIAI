"""Tests for plot_slice and plot_montage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from miai_visualization.exceptions import VisualizationError
from miai_visualization.slices import (
    PlotMontageConfig,
    PlotSliceConfig,
    plot_montage,
    plot_slice,
)


def test_plot_slice_writes_file(tmp_path: Path) -> None:
    volume = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)

    out_path = plot_slice(volume, str(tmp_path / "slice.png"))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_slice_with_mask_writes_file(tmp_path: Path) -> None:
    volume = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[3:5, 3:5, 3:5] = 1

    out_path = plot_slice(volume, str(tmp_path / "slice.png"), mask=mask)

    assert out_path.exists()


def test_plot_slice_mask_shape_mismatch_raises(tmp_path: Path) -> None:
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros((4, 4, 4), dtype=np.uint8)

    with pytest.raises(VisualizationError):
        plot_slice(volume, str(tmp_path / "slice.png"), mask=mask)


def test_plot_montage_writes_file(tmp_path: Path) -> None:
    volume = np.random.default_rng(0).random((8, 8, 8)).astype(np.float32)

    out_path = plot_montage(volume, str(tmp_path / "montage.png"), PlotMontageConfig(num_slices=4))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_montage_num_slices_less_than_one_raises(tmp_path: Path) -> None:
    volume = np.zeros((8, 8, 8), dtype=np.float32)

    with pytest.raises(VisualizationError):
        plot_montage(volume, str(tmp_path / "montage.png"), PlotMontageConfig(num_slices=0))


def test_plot_slice_zero_size_axis_raises(tmp_path: Path) -> None:
    volume = np.zeros((0, 8, 8), dtype=np.float32)

    with pytest.raises(VisualizationError):
        plot_slice(volume, str(tmp_path / "slice.png"), PlotSliceConfig(axis=0))
