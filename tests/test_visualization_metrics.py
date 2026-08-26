"""Tests for plot_metric_summary."""

from __future__ import annotations

from pathlib import Path

import pytest

from miai_visualization.exceptions import VisualizationError
from miai_visualization.metrics import PlotMetricSummaryConfig, plot_metric_summary


def test_plot_metric_summary_bar_writes_file(tmp_path: Path) -> None:
    values = {"case_0": 0.9, "case_1": 0.85, "case_2": 0.92}

    out_path = plot_metric_summary(values, str(tmp_path / "summary.png"))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_metric_summary_box_writes_file(tmp_path: Path) -> None:
    values = {"group_a": [0.9, 0.85, 0.88], "group_b": [0.7, 0.72]}

    out_path = plot_metric_summary(
        values, str(tmp_path / "summary.png"), PlotMetricSummaryConfig(kind="box")
    )

    assert out_path.exists()


def test_plot_metric_summary_with_title(tmp_path: Path) -> None:
    values = {"case_0": 0.9, "case_1": 0.85}

    out_path = plot_metric_summary(
        values, str(tmp_path / "summary.png"), PlotMetricSummaryConfig(title="Dice by case")
    )

    assert out_path.exists()


def test_plot_metric_summary_rotates_labels_when_more_than_six(tmp_path: Path) -> None:
    values = {f"case_{i}": 0.8 + i * 0.01 for i in range(8)}

    out_path = plot_metric_summary(values, str(tmp_path / "summary.png"))

    assert out_path.exists()


def test_plot_metric_summary_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(VisualizationError):
        plot_metric_summary({}, str(tmp_path / "summary.png"))


def test_plot_metric_summary_bar_with_list_value_raises(tmp_path: Path) -> None:
    values = {"case_0": [0.9, 0.8]}

    with pytest.raises(VisualizationError):
        plot_metric_summary(values, str(tmp_path / "summary.png"))
