"""miai-visualization: plotting tools for volumes, comparisons, and metrics.

Everything is saved as a file (matplotlib's non-interactive "Agg"
backend, forced below before any submodule imports
:mod:`matplotlib.pyplot`) rather than shown interactively -- every
visualization is a persisted artifact, consistent with MIAI's
reproducibility-first design (see docs/vision.md). This also means the
package works headlessly (CI runners, this sandbox) without a display
server.

See :mod:`miai_visualization.slices` (single slice / montage),
:mod:`miai_visualization.comparison` (side-by-side + difference maps),
:mod:`miai_visualization.curves` (training curves from a CSV log),
:mod:`miai_visualization.metrics` (bar/box summaries of a metric), and
:mod:`miai_visualization.embeddings` (2D PCA projection of embeddings).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from miai_visualization.comparison import PlotComparisonConfig, plot_comparison  # noqa: E402
from miai_visualization.curves import (  # noqa: E402
    PlotTrainingCurvesConfig,
    plot_training_curves,
)
from miai_visualization.embeddings import (  # noqa: E402
    PlotEmbeddingProjectionConfig,
    plot_embedding_projection,
)
from miai_visualization.exceptions import VisualizationError  # noqa: E402
from miai_visualization.metrics import PlotMetricSummaryConfig, plot_metric_summary  # noqa: E402
from miai_visualization.slices import (  # noqa: E402
    PlotMontageConfig,
    PlotSliceConfig,
    plot_montage,
    plot_slice,
)

__version__ = "0.1.0"

__all__ = [
    "VisualizationError",
    "PlotSliceConfig",
    "plot_slice",
    "PlotMontageConfig",
    "plot_montage",
    "PlotComparisonConfig",
    "plot_comparison",
    "PlotTrainingCurvesConfig",
    "plot_training_curves",
    "PlotMetricSummaryConfig",
    "plot_metric_summary",
    "PlotEmbeddingProjectionConfig",
    "plot_embedding_projection",
]
