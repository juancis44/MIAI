"""MIAI DICOM: reading, writing, metadata, anonymization, and series loading.

Wraps pydicom behind MIAI's exception hierarchy and typed helpers so
downstream packages (miai-pipeline, miai-datasets, ...) interact with
DICOM data through a consistent API instead of calling pydicom directly.
"""

from miai_dicom.anonymize import anonymize
from miai_dicom.exceptions import InvalidDicomFileError
from miai_dicom.io import is_dicom_file, read_dicom, write_dicom
from miai_dicom.metadata import extract_metadata
from miai_dicom.series import DicomSeries, load_series
from miai_dicom.validation import is_valid_dataset, validate_dataset

__version__ = "0.1.0"

__all__ = [
    "anonymize",
    "InvalidDicomFileError",
    "is_dicom_file",
    "read_dicom",
    "write_dicom",
    "extract_metadata",
    "DicomSeries",
    "load_series",
    "is_valid_dataset",
    "validate_dataset",
    "__version__",
]
