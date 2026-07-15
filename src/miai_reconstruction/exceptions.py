"""Exceptions raised by :mod:`miai_reconstruction`."""

from __future__ import annotations

from miai_core.exceptions import MIAIError


class ReconstructionError(MIAIError):
    """Raised for k-space reconstruction configuration or usage errors.

    Examples include an undersampling acceleration factor below 1.0,
    or an empty list of source volumes to reconstruct.
    """
