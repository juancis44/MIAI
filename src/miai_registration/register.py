"""Intensity-based image registration, via SimpleITK's registration framework."""

from __future__ import annotations

from typing import Literal, cast

import SimpleITK as sitk

from miai_core.config import MIAIBaseConfig
from miai_registration.exceptions import RegistrationError

_INTERPOLATORS = {
    "linear": sitk.sitkLinear,
    "nearest": sitk.sitkNearestNeighbor,
    "bspline": sitk.sitkBSpline,
}


class RegistrationConfig(MIAIBaseConfig):
    """Configuration for :func:`register_images`.

    Attributes:
        transform_type: ``"rigid"`` (translation + rotation),
            ``"affine"`` (adds scale + shear), or ``"bspline"``
            (free-form deformable, initialized directly -- for a
            clinical-grade deformable pipeline, register rigidly or
            affinely first and pass the result as a fixed image to a
            second, bspline registration).
        metric: Similarity metric between fixed and moving images.
            ``"mattes_mutual_information"`` (default) works across
            modalities and intensity scales; ``"mean_squares"`` and
            ``"correlation"`` assume the same modality/intensity
            convention as the fixed image.
        interpolator: Used both during optimization and for the final
            resampled image.
        learning_rate: Gradient descent step size.
        number_of_iterations: Maximum optimizer iterations per
            resolution level.
        sampling_percentage: Fraction of voxels randomly sampled per
            iteration to evaluate the metric on (speeds up
            registration at some cost to precision).
        shrink_factors: Multi-resolution pyramid shrink factors, coarse
            to fine (e.g. ``(4, 2, 1)`` registers at quarter, half, then
            full resolution).
        smoothing_sigmas: Gaussian smoothing sigma (in physical units)
            per pyramid level, same length as ``shrink_factors``.
        seed: Random seed for metric sampling, so a run is reproducible.
    """

    transform_type: Literal["rigid", "affine", "bspline"] = "rigid"
    metric: Literal["mean_squares", "mattes_mutual_information", "correlation"] = (
        "mattes_mutual_information"
    )
    interpolator: Literal["linear", "nearest", "bspline"] = "linear"
    learning_rate: float = 1.0
    number_of_iterations: int = 100
    sampling_percentage: float = 0.2
    shrink_factors: tuple[int, ...] = (4, 2, 1)
    smoothing_sigmas: tuple[float, ...] = (2.0, 1.0, 0.0)
    seed: int = 42


def _build_initial_transform(
    fixed: sitk.Image, moving: sitk.Image, transform_type: str
) -> sitk.Transform:
    dim = fixed.GetDimension()
    if transform_type == "rigid":
        rigid_base: sitk.Transform = (
            sitk.Euler3DTransform() if dim == 3 else sitk.Euler2DTransform()
        )
        return cast(
            sitk.Transform,
            sitk.CenteredTransformInitializer(
                fixed, moving, rigid_base, sitk.CenteredTransformInitializerFilter.GEOMETRY
            ),
        )
    if transform_type == "affine":
        affine_base: sitk.Transform = sitk.AffineTransform(dim)
        return cast(
            sitk.Transform,
            sitk.CenteredTransformInitializer(
                fixed, moving, affine_base, sitk.CenteredTransformInitializerFilter.GEOMETRY
            ),
        )
    if transform_type == "bspline":
        mesh_size = [8] * dim
        return cast(sitk.Transform, sitk.BSplineTransformInitializer(fixed, mesh_size, order=3))
    raise RegistrationError(
        f"Unknown transform_type '{transform_type}'. Expected 'rigid', 'affine', or 'bspline'."
    )


def _set_metric(registration_method: sitk.ImageRegistrationMethod, metric: str) -> None:
    if metric == "mean_squares":
        registration_method.SetMetricAsMeanSquares()
    elif metric == "mattes_mutual_information":
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    elif metric == "correlation":
        registration_method.SetMetricAsCorrelation()
    else:
        raise RegistrationError(
            f"Unknown metric '{metric}'. Expected 'mean_squares', "
            "'mattes_mutual_information', or 'correlation'."
        )


def register_images(
    fixed: sitk.Image, moving: sitk.Image, config: RegistrationConfig
) -> tuple[sitk.Transform, sitk.Image]:
    """Register ``moving`` onto ``fixed`` and resample it into the fixed image's grid.

    Uses :class:`SimpleITK.ImageRegistrationMethod` with a multi-resolution
    gradient descent optimizer -- the standard SimpleITK registration
    recipe (initial transform from image geometry, then iterative
    refinement coarse-to-fine per ``config.shrink_factors``).

    Args:
        fixed: The reference image ``moving`` is aligned to.
        moving: The image to align.
        config: Registration parameters.

    Returns:
        ``(transform, resampled_moving)`` -- the estimated transform
        (mapping points in ``fixed`` space to ``moving`` space, per
        ITK's convention) and ``moving`` resampled onto ``fixed``'s
        grid using that transform.

    Raises:
        RegistrationError: If ``config.transform_type`` or
            ``config.metric`` is not recognized.
    """
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    initial_transform = _build_initial_transform(fixed_f, moving_f, config.transform_type)

    registration_method = sitk.ImageRegistrationMethod()
    _set_metric(registration_method, config.metric)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(config.sampling_percentage, seed=config.seed)
    registration_method.SetInterpolator(_INTERPOLATORS[config.interpolator])
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=config.learning_rate,
        numberOfIterations=config.number_of_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=list(config.shrink_factors))
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=list(config.smoothing_sigmas))
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration_method.Execute(fixed_f, moving_f)

    resampled = sitk.Resample(
        moving,
        fixed,
        final_transform,
        _INTERPOLATORS[config.interpolator],
        0.0,
        moving.GetPixelID(),
    )
    return final_transform, resampled
