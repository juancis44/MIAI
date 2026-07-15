"""Exceptions specific to image registration."""

from __future__ import annotations

from miai_core.exceptions import MIAIError


class RegistrationError(MIAIError):
    """Raised when a registration configuration or run is invalid.

    Examples include an unknown transform type or metric, or a
    registration run that fails to produce a usable transform.
    """
