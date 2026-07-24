"""Summarizing a per-case metric (e.g. Dice, PSNR) as a bar or box plot.

Distinct from :mod:`miai_evaluation.metrics` (which *computes*
segmentation metrics) and :mod:`miai_reconstruction.metrics` (which
*computes* reconstruction-quality metrics): this module only
*visualizes* already-computed values, regardless of which package
produced them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt

from miai_core.config import MIAIBaseConfig
from miai_visualization.exceptions import VisualizationError


class PlotMetricSummaryConfig(MIAIBaseConfig):
    """Configuration for :func:`plot_metric_summary`.

    Attributes:
        kind: ``"bar"`` plots one bar per label (each value must be a
            single number); ``"box"`` plots one box per label (each
            value may be a list of numbers, e.g. per-case scores for a
            group, or a single number treated as a one-point sample).
        figsize: Figure size in inches, ``(width, height)``.
        dpi: Output resolution.
        title: Optional plot title.
        ylabel: Y-axis label.
    """

    kind: Literal["bar", "box"] = "bar"
    figsize: tuple[float, float] = (7.0, 4.5)
    dpi: int = 100
    title: str | None = None
    ylabel: str | None = None


def plot_metric_summary(
    values: Mapping[str, float | list[float]],
    output_path: str,
    config: PlotMetricSummaryConfig | None = None,
) -> Path:
    """Plot a bar or box summary of a metric across labeled groups/cases.

    Args:
        values: Label -> value(s). For ``kind="bar"``, every value
            must be a single number. For ``kind="box"``, a value may
            be a list of numbers (a distribution) or a single number.
        output_path: Where the PNG is written. Parent directories are
            created if missing.
        config: Plotting parameters. Uses defaults if ``None``.

    Returns:
        ``output_path`` as a :class:`pathlib.Path`.

    Raises:
        VisualizationError: If ``values`` is empty, or ``kind="bar"``
            is used with a list-valued entry.
    """
    config = config or PlotMetricSummaryConfig()
    if not values:
        raise VisualizationError("values is empty; nothing to plot.")

    labels = list(values.keys())
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    if config.kind == "bar":
        heights: list[float] = []
        for label, value in values.items():
            if isinstance(value, list):
                raise VisualizationError(
                    f"kind='bar' expects one scalar value per label; '{label}' has a list. "
                    "Use kind='box' for distributions."
                )
            heights.append(float(value))
        ax.bar(labels, heights)
    else:
        data = [value if isinstance(value, list) else [float(value)] for value in values.values()]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)

    ax.set_ylabel(config.ylabel or "value")
    if config.title:
        ax.set_title(config.title)
    if len(labels) > 6:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
