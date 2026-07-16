"""Side-by-side comparison plots, e.g. original vs. reconstructed/denoised."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from miai_core.config import MIAIBaseConfig
from miai_visualization.exceptions import VisualizationError


class PlotComparisonConfig(MIAIBaseConfig):
    """Configuration for :func:`plot_comparison`.

    Attributes:
        axis: Which array axis to take a slice along (see
            :attr:`~miai_visualization.slices.PlotSliceConfig.axis`).
        index: Which index along ``axis`` to plot. ``None`` (default)
            plots the middle slice.
        cmap: Colormap for each compared image.
        include_difference_map: If ``True``, appends one extra panel
            per non-reference image showing its absolute difference
            from the first (reference) image.
        diff_cmap: Colormap for difference-map panels.
        subplot_size: Size of each individual panel in inches,
            ``(width, height)``.
        dpi: Output resolution.
    """

    axis: int = 0
    index: int | None = None
    cmap: str = "gray"
    include_difference_map: bool = True
    diff_cmap: str = "inferno"
    subplot_size: tuple[float, float] = (4.0, 4.0)
    dpi: int = 100


def plot_comparison(
    images: dict[str, npt.NDArray[Any]],
    output_path: str,
    config: PlotComparisonConfig | None = None,
) -> Path:
    """Plot the same slice from several volumes side by side.

    The first entry of ``images`` is treated as the reference; if
    ``config.include_difference_map`` is set, an extra panel is added
    per remaining entry showing its absolute difference from the
    reference (e.g. to visualize reconstruction or denoising error).

    Args:
        images: Ordered mapping of label -> volume, all the same
            shape, e.g. ``{"original": ..., "reconstructed": ...}``.
        output_path: Where the PNG is written. Parent directories are
            created if missing.
        config: Plotting parameters. Uses defaults if ``None``.

    Returns:
        ``output_path`` as a :class:`pathlib.Path`.

    Raises:
        VisualizationError: If fewer than 2 images are given, or their
            shapes do not all match.
    """
    config = config or PlotComparisonConfig()
    if len(images) < 2:
        raise VisualizationError("plot_comparison needs at least 2 images to compare.")

    labels = list(images.keys())
    reference = images[labels[0]]
    for label, volume in images.items():
        if volume.shape != reference.shape:
            raise VisualizationError(
                f"Image '{label}' has shape {volume.shape}, expected {reference.shape} "
                f"(matching '{labels[0]}')."
            )

    size = reference.shape[config.axis]
    index = config.index if config.index is not None else size // 2

    diff_labels = labels[1:] if config.include_difference_map else []
    num_panels = len(labels) + len(diff_labels)

    fig, axes = plt.subplots(
        1,
        num_panels,
        figsize=(config.subplot_size[0] * num_panels, config.subplot_size[1]),
        dpi=config.dpi,
    )
    flat_axes = np.atleast_1d(axes).ravel()

    for ax, label in zip(flat_axes[: len(labels)], labels, strict=True):
        image_slice = np.take(images[label], indices=index, axis=config.axis)
        ax.imshow(image_slice, cmap=config.cmap)
        ax.set_title(label)
        ax.axis("off")

    reference_slice = np.take(reference, indices=index, axis=config.axis)
    for ax, label in zip(flat_axes[len(labels) :], diff_labels, strict=True):
        other_slice = np.take(images[label], indices=index, axis=config.axis)
        diff = np.abs(other_slice.astype(np.float64) - reference_slice.astype(np.float64))
        ax.imshow(diff, cmap=config.diff_cmap)
        ax.set_title(f"|{label} - {labels[0]}|")
        ax.axis("off")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
