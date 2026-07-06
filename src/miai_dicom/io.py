"""Reading and writing DICOM files.

Wraps :mod:`pydicom` so that read/write failures raise MIAI's exception
hierarchy (:mod:`miai_core.exceptions`) instead of pydicom's own
exception types, keeping error handling consistent with the rest of the
ecosystem.
"""

from __future__ import annotations

from pathlib import Path

import pydicom
from pydicom.errors import InvalidDicomError

from miai_core.exceptions import NotFoundError
from miai_core.typing import StrPath
from miai_dicom.exceptions import InvalidDicomFileError


def read_dicom(path: StrPath, *, force: bool = False) -> pydicom.Dataset:
    """Read a single DICOM file.

    Args:
        path: Path to a ``.dcm`` file (or any file containing a DICOM
            dataset).
        force: If ``True``, attempt to read the file even if it is
            missing the standard DICOM preamble/magic number (some
            scanners omit it). Passed through to
            :func:`pydicom.dcmread`.

    Returns:
        The parsed :class:`pydicom.Dataset`.

    Raises:
        NotFoundError: If ``path`` does not exist.
        InvalidDicomFileError: If the file cannot be parsed as DICOM.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise NotFoundError(f"DICOM file not found: {file_path}")
    try:
        return pydicom.dcmread(str(file_path), force=force)
    except InvalidDicomError as exc:
        raise InvalidDicomFileError(f"Not a valid DICOM file: {file_path}") from exc


def write_dicom(dataset: pydicom.Dataset, path: StrPath) -> Path:
    """Write a DICOM dataset to disk, creating parent directories.

    Args:
        dataset: The dataset to write. Must have ``file_meta`` set (e.g.
            a dataset returned by :func:`read_dicom`, or one constructed
            with a valid ``file_meta``).
        path: Destination path.

    Returns:
        The destination path.

    Raises:
        InvalidDicomFileError: If ``dataset`` cannot be saved (e.g.
            missing required file meta information).
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset.save_as(str(file_path))
    except Exception as exc:  # pydicom raises varied error types here
        raise InvalidDicomFileError(f"Failed to write DICOM file {file_path}: {exc}") from exc
    return file_path


def is_dicom_file(path: StrPath) -> bool:
    """Check whether a file can be read as a DICOM dataset.

    Args:
        path: Path to check.

    Returns:
        ``True`` if the file exists and parses as DICOM (metadata only,
        pixel data is not decoded), ``False`` otherwise.
    """
    file_path = Path(path)
    if not file_path.exists():
        return False
    try:
        pydicom.dcmread(str(file_path), stop_before_pixels=True)
    except Exception:
        return False
    return True
