"""Apply a previously computed transform to another image."""

from __future__ import annotations

from typing import Literal

import SimpleITK as sitk

_INTERPOLATORS = {
    "linear": sitk.sitkLinear,
    "nearest": sitk.sitkNearestNeighbor,
    "bspline": sitk.sitkBSpline,
}


def apply_transform(
    moving_image: sitk.Image,
    fixed_reference: sitk.Image,
    transform: sitk.Transform,
    interpolator: Literal["linear", "nearest", "bspline"] = "nearest",
    default_value: float = 0.0,
) -> sitk.Image:
    """Resample ``moving_image`` onto ``fixed_reference``'s grid using ``transform``.

    Used to propagate a transform estimated by
    :func:`miai_registration.register.register_images` on an intensity
    image to a paired label mask, so the mask lands in the same space
    the registered image does. Use ``interpolator="nearest"`` (the
    default) for label masks, to avoid inventing fractional label
    values; use ``"linear"`` or ``"bspline"`` for another intensity
    image.

    Args:
        moving_image: The image to resample (e.g. a label mask paired
            with the moving image that was registered).
        fixed_reference: Defines the output grid (size, spacing,
            origin, direction) -- typically the same fixed image
            ``transform`` was computed against.
        transform: A transform from
            :func:`miai_registration.register.register_images` (or
            loaded via
            :func:`miai_registration.transform_io.read_transform`).
        interpolator: Resampling interpolator.
        default_value: Fill value for regions of the output grid that
            fall outside ``moving_image``.

    Returns:
        ``moving_image`` resampled onto ``fixed_reference``'s grid.
    """
    return sitk.Resample(
        moving_image,
        fixed_reference,
        transform,
        _INTERPOLATORS[interpolator],
        default_value,
        moving_image.GetPixelID(),
    )
