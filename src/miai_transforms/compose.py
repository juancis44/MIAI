"""Builds a composed MONAI transform pipeline from a TransformConfig."""

from __future__ import annotations

from typing import Any

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
)

from miai_transforms.config import TransformConfig, TransformSpec
from miai_transforms.exceptions import TransformError
from miai_transforms.sitk_transforms import LoadImageSitkd
from miai_transforms.slice_transforms import ExtractSliced, ExtractSliceStackd

#: Maps a :class:`~miai_transforms.config.TransformSpec.name` to the
#: transform class it builds. ``"load_image"`` uses MIAI's own
#: SimpleITK-backed :class:`~miai_transforms.sitk_transforms.LoadImageSitkd`
#: rather than MONAI's ``LoadImaged`` (which requires an extra reader
#: backend such as nibabel or itk that MIAI does not depend on);
#: ``"extract_slice"``/``"extract_slice_stack"`` are MIAI's own
#: 2D/2.5D slice-reduction transforms (see
#: :mod:`~miai_transforms.slice_transforms`), needed only when
#: ``miai_pipeline.stages.training.TrainingStageConfig.architecture``
#: (or the inference/export equivalent) selects the ``"two_d"`` or
#: ``"two_half_d"`` modality; the rest are MONAI's own array/tensor-only
#: transforms, which need no reader-specific metadata to work. Spatial
#: resampling/orientation is intentionally not offered here -- it's
#: handled upstream by
#: :class:`~miai_pipeline.stages.preprocessing.PreprocessingStage` via
#: SimpleITK. Only the subset needed by MIAI's reference segmentation
#: workflow is registered here; extend as new pipelines need more.
TRANSFORM_REGISTRY: dict[str, type[Any]] = {
    "load_image": LoadImageSitkd,
    "extract_slice": ExtractSliced,
    "extract_slice_stack": ExtractSliceStackd,
    "scale_intensity_range": ScaleIntensityRanged,
    "crop_foreground": CropForegroundd,
    "rand_crop_by_pos_neg_label": RandCropByPosNegLabeld,
    "rand_flip": RandFlipd,
    "rand_rotate90": RandRotate90d,
    "rand_shift_intensity": RandShiftIntensityd,
    "ensure_type": EnsureTyped,
}


def _build_one(spec: TransformSpec) -> Any:
    transform_cls = TRANSFORM_REGISTRY.get(spec.name)
    if transform_cls is None:
        available = ", ".join(sorted(TRANSFORM_REGISTRY))
        raise TransformError(f"Unknown transform '{spec.name}'. Available transforms: {available}.")
    try:
        return transform_cls(**spec.params)
    except TypeError as exc:
        raise TransformError(f"Invalid parameters for transform '{spec.name}': {exc}") from exc


def build_transforms(config: TransformConfig) -> Compose:
    """Build a composed transform pipeline from a config.

    Args:
        config: The ordered list of transform specs to compose.

    Returns:
        A :class:`monai.transforms.Compose` applying each transform in
        ``config.transforms``, in order.

    Raises:
        TransformError: If a transform name is not registered in
            :data:`TRANSFORM_REGISTRY`, or its parameters do not match
            the underlying transform's constructor.
    """
    return Compose([_build_one(spec) for spec in config.transforms])
