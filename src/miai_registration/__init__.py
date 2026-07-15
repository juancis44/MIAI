"""MIAI Registration: intensity-based image registration on SimpleITK.

Provides :func:`~miai_registration.register.register_images` (rigid,
affine, or bspline registration via
:class:`SimpleITK.ImageRegistrationMethod`),
:func:`~miai_registration.apply.apply_transform` (propagating a
computed transform to a paired image, e.g. a label mask), and
:mod:`~miai_registration.transform_io` for saving/loading transforms.
Used by :class:`~miai_pipeline.stages.registration.RegistrationStage`
to align cases to a common reference image within the pipeline.
"""

from miai_registration.apply import apply_transform
from miai_registration.exceptions import RegistrationError
from miai_registration.register import RegistrationConfig, register_images
from miai_registration.transform_io import read_transform, write_transform

__version__ = "0.1.0"

__all__ = [
    "register_images",
    "RegistrationConfig",
    "apply_transform",
    "read_transform",
    "write_transform",
    "RegistrationError",
    "__version__",
]
