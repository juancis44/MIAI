"""Exceptions specific to DICOM handling.

These subclass the shared exceptions in :mod:`miai_core.exceptions` so
callers can choose to catch DICOM-specific failures or the broader MIAI
category, depending on how much detail they need.
"""

from __future__ import annotations

from miai_core.exceptions import MIAIIOError


class InvalidDicomFileError(MIAIIOError):
    """Raised when a file cannot be parsed as a valid DICOM dataset.

    This is distinct from :class:`miai_core.exceptions.NotFoundError`,
    which is raised when the file does not exist at all.
    """
