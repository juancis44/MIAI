"""Exceptions raised by :mod:`miai_visualization`."""

from __future__ import annotations

from miai_core.exceptions import MIAIError


class VisualizationError(MIAIError):
    """Raised for plotting configuration or usage errors.

    Examples include mismatched image shapes in a comparison plot, a
    training log missing an "epoch" column, or too few points/
    dimensions to compute a 2D embedding projection.
    """
