"""MIAI Segmentation: reference segmentation on top of MONAI.

Organized by imaging modality, one subpackage per modality, all three
implemented:

- :mod:`miai_segmentation.three_d` -- full-volume architectures (UNet,
  SegResNet), config-driven training and sliding-window inference.
- :mod:`miai_segmentation.two_d` -- per-slice 2D architectures (UNet,
  AttentionUnet).
- :mod:`miai_segmentation.two_half_d` -- 2.5D (stacked-adjacent-slice)
  architecture (a 2D UNet whose input channels are adjacent slices).

Each modality exposes its own architecture configs and a
``build_model`` dispatcher (see :mod:`miai_segmentation.three_d.models`
for the pattern all three follow). Import a modality's API directly
from its subpackage, e.g. ``from miai_segmentation.three_d import
build_model``. :class:`SegmentationError` is the one exception shared
across all modalities and is re-exported here for convenience.

All three modalities are wired into
:class:`~miai_pipeline.stages.training.TrainingStage`,
:class:`~miai_pipeline.stages.inference.InferenceStage`, and
:class:`~miai_pipeline.stages.export.ExportStage`, via
:mod:`miai_segmentation.modality`'s ``modality``-selecting config
(internal glue, not re-exported here -- see that module's docstring).
Set ``architecture.modality`` to ``"three_d"`` (default), ``"two_d"``,
or ``"two_half_d"`` in a stage's YAML config to pick one; see
`docs/user_guide.md` for a 2D/2.5D example, including the
``extract_slice``/``extract_slice_stack`` transforms 2D/2.5D need.

This root ``__init__.py`` deliberately does **not** re-export
modality-specific names (``UNetConfig``, ``build_model``, ...) -- see
docs/api_design.md's "Package public surface" section for why: a package
gets a documented sub-namespace per mutually-exclusive variant (here, per
modality) specifically to avoid name collisions between them, mirroring
:mod:`miai_pipeline`'s root/``.stages`` split. Each modality subpackage's
own ``__init__.py`` is the public API for that modality.
"""

from miai_segmentation.exceptions import SegmentationError

__version__ = "0.4.0"

__all__ = [
    "SegmentationError",
    "__version__",
]
