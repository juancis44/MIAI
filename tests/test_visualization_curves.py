"""Tests for plot_training_curves."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from miai_visualization.curves import PlotTrainingCurvesConfig, plot_training_curves
from miai_visualization.exceptions import VisualizationError


def _write_log(path: Path) -> Path:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_dice"])
        writer.writeheader()
        for epoch in range(5):
            writer.writerow(
                {"epoch": epoch, "train_loss": 1.0 / (epoch + 1), "val_dice": epoch * 0.1}
            )
    return path


def test_plot_training_curves_writes_file(tmp_path: Path) -> None:
    log_path = _write_log(tmp_path / "log.csv")

    out_path = plot_training_curves(str(log_path), str(tmp_path / "curves.png"))

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_training_curves_selected_metric_only(tmp_path: Path) -> None:
    log_path = _write_log(tmp_path / "log.csv")

    out_path = plot_training_curves(
        str(log_path),
        str(tmp_path / "curves.png"),
        PlotTrainingCurvesConfig(metrics=["val_dice"]),
    )

    assert out_path.exists()


def test_plot_training_curves_empty_log_raises(tmp_path: Path) -> None:
    log_path = tmp_path / "empty.csv"
    with open(log_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss"])
        writer.writeheader()

    with pytest.raises(VisualizationError):
        plot_training_curves(str(log_path), str(tmp_path / "curves.png"))


def test_plot_training_curves_missing_epoch_column_raises(tmp_path: Path) -> None:
    log_path = tmp_path / "bad.csv"
    with open(log_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_loss"])
        writer.writeheader()
        writer.writerow({"step": 0, "train_loss": 1.0})

    with pytest.raises(VisualizationError):
        plot_training_curves(str(log_path), str(tmp_path / "curves.png"))


def test_plot_training_curves_unknown_metric_raises(tmp_path: Path) -> None:
    log_path = _write_log(tmp_path / "log.csv")

    with pytest.raises(VisualizationError):
        plot_training_curves(
            str(log_path),
            str(tmp_path / "curves.png"),
            PlotTrainingCurvesConfig(metrics=["not_a_real_column"]),
        )
