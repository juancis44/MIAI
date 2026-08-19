"""Sliding-window inference for a trained 2.5D segmentation model.

Re-exports :mod:`miai_segmentation.two_d.infer` unchanged. Once a
stacked-slice input tensor is assembled, inference over it is spatially
a 2D sliding window exactly like :mod:`miai_segmentation.two_d`'s --
the channel axis (which slices are stacked) is opaque to
``sliding_window_inference`` and to the mask-thresholding/write-back
logic, only the model's ``in_channels`` (set via
:class:`~miai_segmentation.two_half_d.models.StackedUNetConfig`) needs
to match what was stacked. Kept as a re-export rather than a fresh
duplicate implementation for the same reason
:mod:`miai_segmentation.two_d.train` is: nothing here is actually
2.5D-specific.
"""

from __future__ import annotations

from miai_segmentation.two_d.infer import InferenceConfig, run_case_inference, run_inference

__all__ = [
    "InferenceConfig",
    "run_inference",
    "run_case_inference",
]
