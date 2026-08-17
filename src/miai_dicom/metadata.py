"""Extract a consistent, JSON-serializable metadata dictionary from a DICOM dataset.

Downstream packages (dataset indexing, series loading, clinical pipelines)
should read metadata through :func:`extract_metadata` rather than
accessing pydicom attributes directly, so a change in which tags are
considered "core" only needs to happen in one place.
"""

from __future__ import annotations

from typing import Any

import pydicom

from miai_core.typing import JSONDict

#: Mapping of friendly metadata keys to their DICOM keyword.
CORE_TAGS: dict[str, str] = {
    "patient_id": "PatientID",
    "patient_sex": "PatientSex",
    "patient_birth_date": "PatientBirthDate",
    "study_instance_uid": "StudyInstanceUID",
    "series_instance_uid": "SeriesInstanceUID",
    "sop_instance_uid": "SOPInstanceUID",
    "modality": "Modality",
    "manufacturer": "Manufacturer",
    "rows": "Rows",
    "columns": "Columns",
    "pixel_spacing": "PixelSpacing",
    "slice_thickness": "SliceThickness",
    "instance_number": "InstanceNumber",
    "study_date": "StudyDate",
    "series_description": "SeriesDescription",
}


def extract_metadata(dataset: pydicom.Dataset, *, include_patient_name: bool = False) -> JSONDict:
    """Extract a flat, JSON-serializable metadata dictionary from a dataset.

    Args:
        dataset: A DICOM dataset, e.g. from
            :func:`miai_dicom.io.read_dicom`.
        include_patient_name: Whether to include ``patient_name``. Off by
            default since patient name is direct PHI and callers should
            opt in deliberately (see :mod:`miai_dicom.anonymize`).

    Returns:
        A dictionary keyed by the friendly names in :data:`CORE_TAGS`
        (plus ``patient_name`` if requested). Missing tags are included
        with a value of ``None`` rather than omitted, so downstream code
        can rely on a stable set of keys.
    """
    metadata: JSONDict = {
        key: _to_jsonable(getattr(dataset, tag, None)) for key, tag in CORE_TAGS.items()
    }
    if include_patient_name:
        metadata["patient_name"] = _to_jsonable(getattr(dataset, "PatientName", None))
    return metadata


def _to_jsonable(value: Any) -> Any:
    """Convert a pydicom value representation to a plain JSON-safe type."""
    if value is None:
        return None
    if isinstance(value, pydicom.multival.MultiValue):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, pydicom.valuerep.PersonName):
        return str(value)
    if isinstance(value, (int, float, str)):
        return value
    return str(value)
