"""Exceptions specific to segmentation model training and inference."""

from __future__ import annotations

from miai_core.exceptions import MIAIError


class SegmentationError(MIAIError):
    """Raised when training or inference cannot proceed.

    Examples include an empty data loader, or a checkpoint file that
    does not match the configured model architecture.
    """
