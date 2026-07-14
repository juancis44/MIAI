"""MIAI Transforms: config-driven MONAI transform pipelines.

Wraps MONAI's dictionary-based transforms behind a small, named
registry (:data:`miai_transforms.compose.TRANSFORM_REGISTRY`) so a
transform pipeline can be defined entirely from a YAML
:class:`~miai_transforms.config.TransformConfig`, consistent with how
:mod:`miai_pipeline` stages are configured.
"""

from miai_transforms.compose import TRANSFORM_REGISTRY, build_transforms
from miai_transforms.config import TransformConfig, TransformSpec
from miai_transforms.exceptions import TransformError

__version__ = "0.1.0"

__all__ = [
    "build_transforms",
    "TRANSFORM_REGISTRY",
    "TransformConfig",
    "TransformSpec",
    "TransformError",
    "__version__",
]
