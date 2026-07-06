"""Validating that a DICOM dataset contains the tags a workflow needs.

Distinct from :mod:`miai_dicom.io`, which validates that a *file* parses
as DICOM at all: this module validates that a *parsed dataset* carries
the specific tags a given pipeline stage requires (e.g. pixel geometry
before running a segmentation model).
"""

from __future__ import annotations

from collections.abc import Iterable

import pydicom

from miai_core.exceptions import ValidationError

#: Tags required for a dataset to be minimally usable: identifiable,
#: typed by modality, and describing a 2D pixel grid.
REQUIRED_TAGS_DEFAULT: tuple[str, ...] = ("SOPInstanceUID", "Modality", "Rows", "Columns")


def validate_dataset(
    dataset: pydicom.Dataset, *, required_tags: Iterable[str] = REQUIRED_TAGS_DEFAULT
) -> None:
    """Raise if ``dataset`` is missing any of ``required_tags``.

    Args:
        dataset: The dataset to validate.
        required_tags: DICOM keywords that must be present. Defaults to
            :data:`REQUIRED_TAGS_DEFAULT`.

    Raises:
        ValidationError: If one or more ``required_tags`` are missing,
            naming all of them in the error message.
    """
    missing = [tag for tag in required_tags if tag not in dataset]
    if missing:
        raise ValidationError(f"DICOM dataset is missing required tag(s): {', '.join(missing)}")


def is_valid_dataset(
    dataset: pydicom.Dataset, *, required_tags: Iterable[str] = REQUIRED_TAGS_DEFAULT
) -> bool:
    """Return whether ``dataset`` satisfies :func:`validate_dataset`.

    Args:
        dataset: The dataset to check.
        required_tags: See :func:`validate_dataset`.

    Returns:
        ``True`` if validation passes, ``False`` otherwise.
    """
    try:
        validate_dataset(dataset, required_tags=required_tags)
    except ValidationError:
        return False
    return True
