"""Exceptions specific to diffusion models."""

from __future__ import annotations

from miai_core.exceptions import MIAIError


class DiffusionError(MIAIError):
    """Raised when a diffusion schedule or model configuration is invalid.

    Examples include an unknown noise schedule name, or a denoising
    start timestep outside the schedule's range.
    """
