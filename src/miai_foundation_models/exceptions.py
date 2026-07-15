"""Exceptions raised by :mod:`miai_foundation_models`."""

from __future__ import annotations

from miai_core.exceptions import MIAIError


class FoundationModelError(MIAIError):
    """Raised for foundation-model configuration or usage errors.

    Examples include an empty volume passed for feature extraction, or
    an unknown pooling strategy reaching runtime code (bypassing
    pydantic's own validation, e.g. via ``model_copy(update=...)``).
    """
