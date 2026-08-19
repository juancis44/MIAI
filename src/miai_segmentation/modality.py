"""Cross-modality dispatch, for :mod:`miai_pipeline.stages` to use.

Internal glue -- not re-exported from :mod:`miai_segmentation`'s root
``__init__.py``, per `docs/api_design.md`'s "Package public surface"
section (it is not a modality's own public API; it exists only so
:class:`~miai_pipeline.stages.training.TrainingStage`,
:class:`~miai_pipeline.stages.inference.InferenceStage`, and
:class:`~miai_pipeline.stages.export.ExportStage` can select *which*
modality (:mod:`~miai_segmentation.three_d`,
:mod:`~miai_segmentation.two_d`, or
:mod:`~miai_segmentation.two_half_d`) an experiment uses from one config
field, instead of hardcoding one).

Each modality's ``ArchitectureConfig`` is a distinct type (they are not
interchangeable -- ``two_d``'s has an ``attention_unet`` option
``three_d``'s doesn't, for instance), so
:class:`SegmentationModalityConfig` declares one nested field per
modality (all three always present, only the ``modality``-selected one
actually used) rather than trying to type ``architecture`` as a single
field that changes shape -- the same "kind selects one of several
declared-but-mostly-unused nested configs" pattern each modality's own
``ArchitectureConfig`` already uses one level down.
:class:`SegmentationInferenceConfig` does the same for sliding-window
inference parameters, which only :class:`~miai_pipeline.stages.
inference.InferenceStageConfig` needs -- ``two_half_d`` has no field of
its own here because it re-exports ``two_d``'s ``InferenceConfig``
unchanged (see :mod:`miai_segmentation.two_half_d.infer`).
"""

from __future__ import annotations

from typing import Literal

import torch

from miai_core.config import MIAIBaseConfig
from miai_segmentation.exceptions import SegmentationError
from miai_segmentation.three_d.infer import InferenceConfig as ThreeDInferenceConfig
from miai_segmentation.three_d.models import ArchitectureConfig as ThreeDArchitectureConfig
from miai_segmentation.three_d.models import build_model as _build_three_d_model
from miai_segmentation.two_d.infer import InferenceConfig as TwoDInferenceConfig
from miai_segmentation.two_d.models import ArchitectureConfig as TwoDArchitectureConfig
from miai_segmentation.two_d.models import build_model as _build_two_d_model
from miai_segmentation.two_half_d.models import ArchitectureConfig as TwoHalfDArchitectureConfig
from miai_segmentation.two_half_d.models import build_model as _build_two_half_d_model

#: The three implemented segmentation modalities. Kept as one literal
#: (rather than each stage config redeclaring it) so a new modality only
#: needs to be added here and to :class:`SegmentationModalityConfig`.
Modality = Literal["three_d", "two_d", "two_half_d"]


class SegmentationModalityConfig(MIAIBaseConfig):
    """Selects a segmentation modality and its architecture.

    Attributes:
        modality: Which modality :func:`build_model_for_modality`
            builds a model for.
        three_d: Used when ``modality == "three_d"``.
        two_d: Used when ``modality == "two_d"``.
        two_half_d: Used when ``modality == "two_half_d"``.
    """

    modality: Modality = "three_d"
    three_d: ThreeDArchitectureConfig = ThreeDArchitectureConfig()
    two_d: TwoDArchitectureConfig = TwoDArchitectureConfig()
    two_half_d: TwoHalfDArchitectureConfig = TwoHalfDArchitectureConfig()


def build_model_for_modality(config: SegmentationModalityConfig) -> torch.nn.Module:
    """Build the model selected by ``config.modality``.

    Args:
        config: The modality selection and its per-modality settings.

    Returns:
        An uninitialized (freshly constructed) model for the selected
        modality/architecture.

    Raises:
        SegmentationError: If ``config.modality`` is not a recognized
            modality. Not reachable through normal use --
            ``modality`` is a :data:`Modality` literal validated by
            Pydantic -- but guards against a config bypassing that
            validation (e.g. constructed via ``model_construct``).
    """
    if config.modality == "three_d":
        return _build_three_d_model(config.three_d)
    if config.modality == "two_d":
        return _build_two_d_model(config.two_d)
    if config.modality == "two_half_d":
        return _build_two_half_d_model(config.two_half_d)
    raise SegmentationError(f"Unknown segmentation modality: {config.modality!r}")


class SegmentationInferenceConfig(MIAIBaseConfig):
    """Selects modality-appropriate sliding-window inference parameters.

    Attributes:
        three_d: Used when the owning stage's modality is ``"three_d"``
            (``roi_size`` is a 3-tuple).
        two_d: Used when the owning stage's modality is ``"two_d"`` or
            ``"two_half_d"`` (``roi_size`` is a 2-tuple -- both
            modalities run a 2D sliding window; see
            :mod:`miai_segmentation.two_half_d.infer`).
    """

    three_d: ThreeDInferenceConfig = ThreeDInferenceConfig()
    two_d: TwoDInferenceConfig = TwoDInferenceConfig()


def inference_config_for_modality(
    modality: Modality, config: SegmentationInferenceConfig
) -> ThreeDInferenceConfig | TwoDInferenceConfig:
    """Select the sliding-window inference config matching ``modality``.

    Args:
        modality: Which modality is running inference.
        config: The modality-keyed inference parameter bundle.

    Returns:
        ``config.three_d`` for ``"three_d"``; ``config.two_d`` for both
        ``"two_d"`` and ``"two_half_d"`` (they share one 2D-window
        config shape).

    Raises:
        SegmentationError: If ``modality`` is not a recognized modality.
    """
    if modality == "three_d":
        return config.three_d
    if modality in ("two_d", "two_half_d"):
        return config.two_d
    raise SegmentationError(f"Unknown segmentation modality: {modality!r}")
