"""Exceptions specific to dataset construction."""

from __future__ import annotations

from miai_core.exceptions import MIAIError


class DatasetBuildError(MIAIError):
    """Raised when a manifest split cannot be turned into a MONAI dataset.

    Examples include a manifest entry missing the required ``"image"``
    key, or an empty split passed where at least one case is required.
    """
