"""Builds a composed MONAI transform pipeline from a TransformConfig."""

from __future__ import annotations

from typing import Any

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

from miai_transforms.config import TransformConfig, TransformSpec
from miai_transforms.exceptions import TransformError

#: Maps a :class:`~miai_transforms.config.TransformSpec.name` to the
#: MONAI dictionary-based transform class it builds. Only the subset of
#: MONAI's transforms needed by MIAI's reference segmentation workflow
#: is registered here; extend as new pipelines need more transforms.
TRANSFORM_REGISTRY: dict[str, type[Any]] = {
    "load_image": LoadImaged,
    "ensure_channel_first": EnsureChannelFirstd,
    "orientation": Orientationd,
    "spacing": Spacingd,
    "scale_intensity_range": ScaleIntensityRanged,
    "normalize_intensity": NormalizeIntensityd,
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
    """Build a composed MONAI transform pipeline from a config.

    Args:
        config: The ordered list of transform specs to compose.

    Returns:
        A :class:`monai.transforms.Compose` applying each transform in
        ``config.transforms``, in order.

    Raises:
        TransformError: If a transform name is not registered in
            :data:`TRANSFORM_REGISTRY`, or its parameters do not match
            the underlying MONAI transform's constructor.
    """
    return Compose([_build_one(spec) for spec in config.transforms])
