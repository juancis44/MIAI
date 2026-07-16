"""Plotting a single slice or a montage of slices from a volume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from miai_core.config import MIAIBaseConfig
from miai_visualization.exceptions import VisualizationError


class PlotSliceConfig(MIAIBaseConfig):
    """Configuration for :func:`plot_slice`.

    Attributes:
        axis: Which array axis to take a slice along. Volumes follow
            SimpleITK's ``(D, H, W)`` array convention throughout MIAI,
            so ``0`` (the default) takes an axial slice.
        index: Which index along ``axis`` to plot. ``None`` (default)
            plots the middle slice.
        cmap: Colormap for the base image.
        mask_cmap: Colormap for the overlaid mask, if one is given.
        mask_alpha: Opacity of the overlaid mask.
        figsize: Figure size in inches, ``(width, height)``.
        dpi: Output resolution.
        title: Optional plot title.
    """

    axis: int = 0
    index: int | None = None
    cmap: str = "gray"
    mask_cmap: str = "Reds"
    mask_alpha: float = 0.4
    figsize: tuple[float, float] = (6.0, 6.0)
    dpi: int = 100
    title: str | None = None


def _resolve_index(size: int, index: int | None) -> int:
    if size == 0:
        raise VisualizationError("Cannot slice a volume with size 0 along the chosen axis.")
    return index if index is not None else size // 2


def plot_slice(
    volume: npt.NDArray[Any],
    output_path: str,
    config: PlotSliceConfig | None = None,
    mask: npt.NDArray[Any] | None = None,
) -> Path:
    """Plot a single 2D slice of a volume, with an optional mask overlay.

    Args:
        volume: A grayscale volume array, ``(D, H, W)`` convention.
        output_path: Where the PNG is written. Parent directories are
            created if missing.
        config: Plotting parameters. Uses defaults if ``None``.
        mask: An optional binary/label mask, same shape as ``volume``,
            drawn as a semi-transparent overlay (e.g. a segmentation
            prediction or ground truth).

    Returns:
        ``output_path`` as a :class:`pathlib.Path`.

    Raises:
        VisualizationError: If the chosen slice axis has size 0, or
            ``mask`` does not match ``volume``'s shape.
    """
    config = config or PlotSliceConfig()
    if mask is not None and mask.shape != volume.shape:
        raise VisualizationError(
            f"mask shape {mask.shape} does not match volume shape {volume.shape}."
        )

    index = _resolve_index(volume.shape[config.axis], config.index)
    image_slice = np.take(volume, indices=index, axis=config.axis)

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    ax.imshow(image_slice, cmap=config.cmap)
    if mask is not None:
        mask_slice = np.take(mask, indices=index, axis=config.axis)
        masked = np.ma.masked_where(mask_slice == 0, mask_slice)
        ax.imshow(masked, cmap=config.mask_cmap, alpha=config.mask_alpha)
    ax.axis("off")
    if config.title:
        ax.set_title(config.title)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


class PlotMontageConfig(MIAIBaseConfig):
    """Configuration for :func:`plot_montage`.

    Attributes:
        axis: Which array axis to take slices along (see
            :attr:`PlotSliceConfig.axis`).
        num_slices: How many evenly-spaced slices to include.
        cmap: Colormap for each slice.
        subplot_size: Size of each individual slice's panel in inches,
            ``(width, height)``.
        dpi: Output resolution.
        title: Optional figure-level title.
    """

    axis: int = 0
    num_slices: int = 9
    cmap: str = "gray"
    subplot_size: tuple[float, float] = (2.5, 2.5)
    dpi: int = 100
    title: str | None = None


def plot_montage(
    volume: npt.NDArray[Any], output_path: str, config: PlotMontageConfig | None = None
) -> Path:
    """Plot a grid of evenly-spaced slices from a volume.

    Args:
        volume: A grayscale volume array, ``(D, H, W)`` convention.
        output_path: Where the PNG is written. Parent directories are
            created if missing.
        config: Plotting parameters. Uses defaults if ``None``.

    Returns:
        ``output_path`` as a :class:`pathlib.Path`.

    Raises:
        VisualizationError: If ``config.num_slices`` is less than 1,
            or the chosen slice axis has size 0.
    """
    config = config or PlotMontageConfig()
    if config.num_slices < 1:
        raise VisualizationError(f"num_slices must be >= 1, got {config.num_slices}.")

    size = volume.shape[config.axis]
    if size == 0:
        raise VisualizationError("Cannot slice a volume with size 0 along the chosen axis.")

    num_slices = min(config.num_slices, size)
    indices = np.linspace(0, size - 1, num=num_slices, dtype=int)

    num_cols = int(np.ceil(np.sqrt(num_slices)))
    num_rows = int(np.ceil(num_slices / num_cols))

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(config.subplot_size[0] * num_cols, config.subplot_size[1] * num_rows),
        dpi=config.dpi,
    )
    flat_axes = np.atleast_1d(axes).ravel()

    for ax, index in zip(flat_axes, indices, strict=False):
        image_slice = np.take(volume, indices=int(index), axis=config.axis)
        ax.imshow(image_slice, cmap=config.cmap)
        ax.set_title(str(int(index)), fontsize=8)
        ax.axis("off")
    for ax in flat_axes[len(indices) :]:
        ax.axis("off")

    if config.title:
        fig.suptitle(config.title)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
