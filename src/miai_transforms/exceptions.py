"""Exceptions specific to transform composition."""

from __future__ import annotations

from miai_core.exceptions import MIAIError


class TransformError(MIAIError):
    """Raised when a transform spec cannot be resolved or composed.

    Examples include an unknown transform name, or parameters that do
    not match the underlying MONAI transform's signature.
    """
