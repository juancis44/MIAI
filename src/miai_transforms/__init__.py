"""MIAI Transforms: config-driven transform pipelines on MONAI + SimpleITK.

Wraps a small, named registry (:data:`miai_transforms.compose.TRANSFORM_REGISTRY`)
of MONAI's array/tensor transforms plus MIAI's own SimpleITK-backed
image loader (:class:`~miai_transforms.sitk_transforms.LoadImageSitkd`),
so a transform pipeline can be defined entirely from a YAML
:class:`~miai_transforms.config.TransformConfig`, consistent with how
:mod:`miai_pipeline` stages are configured.
"""

from miai_transforms.compose import TRANSFORM_REGISTRY, build_transforms
from miai_transforms.config import TransformConfig, TransformSpec
from miai_transforms.exceptions import TransformError
from miai_transforms.sitk_transforms import LoadImageSitkd

__version__ = "0.1.0"

__all__ = [
    "build_transforms",
    "TRANSFORM_REGISTRY",
    "TransformConfig",
    "TransformSpec",
    "TransformError",
    "LoadImageSitkd",
    "__version__",
]
