"""MIAI Segmentation: reference segmentation on top of MONAI.

Organized by imaging modality, one subpackage per modality:

- :mod:`miai_segmentation.three_d` -- full-volume architectures (UNet,
  SegResNet), config-driven training and sliding-window inference.
  Implemented.
- ``miai_segmentation.two_d`` -- per-slice 2D architectures. Not yet
  implemented; planned as a follow-up to ``three_d``.
- ``miai_segmentation.two_half_d`` -- 2.5D (stacked-adjacent-slice)
  architectures. Not yet implemented; planned as a follow-up to
  ``three_d``.

Each modality is expected to expose its own architecture configs and a
``build_model`` dispatcher (see :mod:`miai_segmentation.three_d.models`),
so :class:`~miai_pipeline.stages.training.TrainingStage` and
:class:`~miai_pipeline.stages.inference.InferenceStage` can select a
modality and architecture entirely from YAML. Import a modality's API
directly from its subpackage, e.g.
``from miai_segmentation.three_d import build_model``.
:class:`SegmentationError` is the one exception shared across all
modalities and is re-exported here for convenience.

This root ``__init__.py`` deliberately does **not** re-export
modality-specific names (``UNetConfig``, ``build_model``, ...) -- see
docs/api_design.md's "Package public surface" section for why: a package
gets a documented sub-namespace per mutually-exclusive variant (here, per
modality) specifically to avoid name collisions between them, mirroring
:mod:`miai_pipeline`'s root/``.stages`` split. Each modality subpackage's
own ``__init__.py`` is the public API for that modality.
"""

from miai_segmentation.exceptions import SegmentationError

__version__ = "0.2.0"

__all__ = [
    "SegmentationError",
    "__version__",
]
