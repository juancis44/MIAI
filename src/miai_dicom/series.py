"""Grouping a directory of DICOM files into series.

Clinical DICOM exports are typically a flat (or nested) directory of
per-slice files with no naming convention that indicates series
membership or slice order. :func:`load_series` groups files by
``SeriesInstanceUID`` and sorts each group into acquisition order, so
downstream code can treat a series as a single ordered volume.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pydicom

from miai_core.exceptions import NotFoundError, ValidationError
from miai_core.typing import StrPath
from miai_dicom.io import is_dicom_file, read_dicom


@dataclass(frozen=True)
class DicomSeries:
    """A single DICOM series: a set of files sharing a SeriesInstanceUID.

    Attributes:
        series_instance_uid: The series' unique identifier.
        modality: The acquisition modality (e.g. ``"CT"``, ``"MR"``), or
            ``None`` if not present on the first file in the series.
        file_paths: Paths to the series' files, sorted into acquisition
            order (by ``InstanceNumber`` when available, falling back to
            the z-component of ``ImagePositionPatient``).
    """

    series_instance_uid: str
    modality: str | None
    file_paths: tuple[Path, ...]

    def __len__(self) -> int:
        return len(self.file_paths)


def load_series(directory: StrPath) -> list[DicomSeries]:
    """Scan a directory for DICOM files and group them into series.

    Args:
        directory: Directory to scan recursively for DICOM files. Files
            that do not parse as DICOM, or that lack a
            ``SeriesInstanceUID``, are skipped.

    Returns:
        One :class:`DicomSeries` per distinct ``SeriesInstanceUID``
        found, each with its files sorted into acquisition order.

    Raises:
        NotFoundError: If ``directory`` does not exist or is not a
            directory.
        ValidationError: If no valid DICOM series are found under
            ``directory``.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotFoundError(f"Directory not found: {dir_path}")

    by_series: dict[str, list[tuple[Path, pydicom.Dataset]]] = {}
    for file_path in _iter_dicom_files(dir_path):
        dataset = read_dicom(file_path, force=True)
        uid = getattr(dataset, "SeriesInstanceUID", None)
        if uid is None:
            continue
        by_series.setdefault(uid, []).append((file_path, dataset))

    if not by_series:
        raise ValidationError(f"No DICOM series found under {dir_path}")

    series_list = []
    for uid, entries in by_series.items():
        entries.sort(key=lambda item: _sort_key(item[1]))
        modality = getattr(entries[0][1], "Modality", None)
        series_list.append(
            DicomSeries(
                series_instance_uid=uid,
                modality=modality,
                file_paths=tuple(path for path, _ in entries),
            )
        )
    return series_list


def _iter_dicom_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and is_dicom_file(path):
            yield path


def _sort_key(dataset: pydicom.Dataset) -> tuple[int, float]:
    instance_number = getattr(dataset, "InstanceNumber", None)
    if instance_number is not None:
        return (0, float(instance_number))

    position = getattr(dataset, "ImagePositionPatient", None)
    if position is not None:
        return (1, float(position[2]))

    return (2, 0.0)
