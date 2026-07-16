"""Plotting a 2D projection of high-dimensional embeddings.

Computes PCA by hand via SVD (:func:`torch.linalg.svd`) rather than
adding scikit-learn as a dependency: MIAI already includes ``torch``,
and a 2-component PCA projection is a small, self-contained linear
algebra operation that does not need a dedicated library.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from miai_core.config import MIAIBaseConfig
from miai_visualization.exceptions import VisualizationError


class PlotEmbeddingProjectionConfig(MIAIBaseConfig):
    """Configuration for :func:`plot_embedding_projection`.

    Attributes:
        figsize: Figure size in inches, ``(width, height)``.
        dpi: Output resolution.
        title: Optional plot title.
        point_size: Marker size for each embedding's point.
        cmap: Colormap used to color points by ``labels`` (see
            :func:`plot_embedding_projection`), when labels are given.
    """

    figsize: tuple[float, float] = (6.0, 6.0)
    dpi: int = 100
    title: str | None = None
    point_size: float = 20.0
    cmap: str = "tab10"


def _pca_2d(embeddings: torch.Tensor) -> torch.Tensor:
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:2]
    return cast(torch.Tensor, centered @ components.T)


def plot_embedding_projection(
    embeddings: torch.Tensor,
    output_path: str,
    config: PlotEmbeddingProjectionConfig | None = None,
    labels: list[str] | None = None,
) -> Path:
    """Plot a 2-component PCA projection of a set of embeddings.

    Args:
        embeddings: Shape ``(N, D)``, one embedding per row (e.g. from
            :func:`miai_foundation_models.extractor.FeatureExtractor`).
        output_path: Where the PNG is written. Parent directories are
            created if missing.
        config: Plotting parameters. Uses defaults if ``None``.
        labels: Optional per-embedding category label (length ``N``),
            used to color points and add a legend.

    Returns:
        ``output_path`` as a :class:`pathlib.Path`.

    Raises:
        VisualizationError: If ``embeddings`` has fewer than 2 rows or
            fewer than 2 columns, or ``labels`` does not have one
            entry per embedding.
    """
    config = config or PlotEmbeddingProjectionConfig()
    if embeddings.ndim != 2 or embeddings.shape[0] < 2 or embeddings.shape[1] < 2:
        raise VisualizationError(
            "embeddings must be 2D with at least 2 rows and 2 columns, "
            f"got shape {tuple(embeddings.shape)}."
        )
    if labels is not None and len(labels) != embeddings.shape[0]:
        raise VisualizationError(
            f"labels has {len(labels)} entries, expected {embeddings.shape[0]} "
            "(one per embedding)."
        )

    projected = _pca_2d(embeddings).cpu().numpy()

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    if labels is None:
        ax.scatter(projected[:, 0], projected[:, 1], s=config.point_size)
    else:
        cmap = plt.get_cmap(config.cmap)
        unique_labels = sorted(set(labels))
        for i, label in enumerate(unique_labels):
            mask = np.array([entry == label for entry in labels])
            ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                s=config.point_size,
                color=cmap(i % cmap.N),
                label=label,
            )
        ax.legend()

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if config.title:
        ax.set_title(config.title)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
