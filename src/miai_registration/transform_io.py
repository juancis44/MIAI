"""Reading and writing SimpleITK transforms."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import SimpleITK as sitk

from miai_core.io import ensure_dir
from miai_core.typing import StrPath


def write_transform(transform: sitk.Transform, path: StrPath) -> Path:
    """Write a transform to disk, creating parent directories.

    Args:
        transform: A transform, e.g. from
            :func:`miai_registration.register.register_images`.
        path: Destination path (SimpleITK infers the file format from
            the extension, e.g. ``.tfm``).

    Returns:
        The destination path.
    """
    out_path = Path(path)
    ensure_dir(out_path.parent)
    sitk.WriteTransform(transform, str(out_path))
    return out_path


def read_transform(path: StrPath) -> sitk.Transform:
    """Read a transform previously written by :func:`write_transform`.

    Args:
        path: Path to a transform file.

    Returns:
        The loaded transform.
    """
    return cast(sitk.Transform, sitk.ReadTransform(str(path)))
