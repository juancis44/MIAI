"""Plotting training curves from a CSV log."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from miai_core.config import MIAIBaseConfig
from miai_visualization.exceptions import VisualizationError


class PlotTrainingCurvesConfig(MIAIBaseConfig):
    """Configuration for :func:`plot_training_curves`.

    Attributes:
        metrics: Which CSV columns to plot as separate lines. ``None``
            (default) plots every column except ``"epoch"``.
        figsize: Figure size in inches, ``(width, height)``.
        dpi: Output resolution.
        title: Optional plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """

    metrics: list[str] | None = None
    figsize: tuple[float, float] = (7.0, 4.5)
    dpi: int = 100
    title: str | None = None
    xlabel: str = "epoch"
    ylabel: str | None = None


def plot_training_curves(
    log_path: str, output_path: str, config: PlotTrainingCurvesConfig | None = None
) -> Path:
    """Plot one line per metric from a CSV training log.

    Args:
        log_path: Path to a CSV file with an ``"epoch"`` column plus
            one column per metric (e.g. ``"train_loss"``,
            ``"val_dice"``).
        output_path: Where the PNG is written. Parent directories are
            created if missing.
        config: Plotting parameters. Uses defaults if ``None``.

    Returns:
        ``output_path`` as a :class:`pathlib.Path`.

    Raises:
        VisualizationError: If the log has no rows, is missing an
            ``"epoch"`` column, or a requested metric column is
            missing.
    """
    config = config or PlotTrainingCurvesConfig()

    with open(log_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise VisualizationError(f"Training log '{log_path}' has no rows.")
    if "epoch" not in rows[0]:
        raise VisualizationError(f"Training log '{log_path}' is missing an 'epoch' column.")

    metric_names = config.metrics or [key for key in rows[0] if key != "epoch"]
    for metric in metric_names:
        if metric not in rows[0]:
            raise VisualizationError(f"Training log '{log_path}' has no column '{metric}'.")

    epochs = [float(row["epoch"]) for row in rows]

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    for metric in metric_names:
        values = [float(row[metric]) for row in rows]
        ax.plot(epochs, values, label=metric)

    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel or "value")
    if config.title:
        ax.set_title(config.title)
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
