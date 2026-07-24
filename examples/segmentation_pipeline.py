"""End-to-end example: the full MIAI segmentation pipeline on synthetic data.

Generates a small synthetic dataset -- multi-slice DICOM series (real
pixel data, no patient information) plus matching NIfTI segmentation
labels -- then runs it through :data:`examples/configs/pipeline.yaml`
end to end via :meth:`~miai_pipeline.pipeline.Pipeline.from_config`:

    DICOM -> NIfTI -> Preprocessing -> Dataset -> Training -> Inference
    -> Evaluation

No real dataset or GPU is required; everything here runs on CPU in
well under a minute. This mirrors (and is more complete than) the
"Quick example" snippets in the root README.md, which show each
package in isolation rather than one pipeline run start to finish.

The data-generation helpers below are intentionally standalone, not
imported from ``tests/conftest.py`` -- that module is test-only
fixture code, not part of MIAI's public API, so an example meant to be
read and copied by users shouldn't depend on it.

Run:
    python examples/segmentation_pipeline.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

from miai_core.logging import configure_logging, get_logger
from miai_pipeline import Pipeline, PipelineConfig, PipelineContext

logger = get_logger(__name__)

EXAMPLE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXAMPLE_DIR / "configs" / "pipeline.yaml"
OUTPUT_DIR = EXAMPLE_DIR / "output"
DICOM_DIR = OUTPUT_DIR / "synthetic_dicom"
LABEL_DIR = OUTPUT_DIR / "synthetic_labels"

#: Number of synthetic cases to generate. With the 0.3/0.3 val/test
#: split in configs/pipeline.yaml, this gives 4 train, 3 val, 3 test.
NUM_CASES = 10

#: (depth, rows, columns), matching the sliding-window `roi_size` in
#: configs/pipeline.yaml's inference stage -- a single window covers
#: the whole volume, keeping the demo fast.
VOLUME_SHAPE = (32, 32, 32)


def _make_labeled_volume(
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Build one synthetic (image, label) volume pair.

    The label is a solid cube roughly centered in the volume, at a
    randomized offset and size per case (so cases aren't identical,
    unlike the fixed pattern ``tests/conftest.py`` uses for
    determinism). The image is that same cube, scaled into a plausible
    CT-like intensity range, plus Gaussian noise -- an easy, real
    signal for the tiny demo UNet in configs/pipeline.yaml to learn
    from in just a few epochs.
    """
    depth, rows, columns = VOLUME_SHAPE
    label = np.zeros(VOLUME_SHAPE, dtype=np.uint8)

    size_frac = rng.uniform(0.35, 0.55)
    d0, d1 = _cube_bounds(depth, size_frac, rng)
    r0, r1 = _cube_bounds(rows, size_frac, rng)
    c0, c1 = _cube_bounds(columns, size_frac, rng)
    label[d0:d1, r0:r1, c0:c1] = 1

    image = label.astype(np.float32) * 500.0
    image += rng.normal(loc=0.0, scale=25.0, size=VOLUME_SHAPE).astype(np.float32)
    return image, label


def _cube_bounds(extent: int, size_frac: float, rng: np.random.Generator) -> tuple[int, int]:
    size = max(2, int(extent * size_frac))
    start = int(rng.uniform(0, extent - size))
    return start, start + size


def _write_dicom_series(directory: Path, image: npt.NDArray[np.float32], case_index: int) -> None:
    """Write ``image`` as a synthetic multi-slice DICOM series.

    Each slice gets its own ``SOPInstanceUID``, sharing one
    ``SeriesInstanceUID``/``StudyInstanceUID`` for the whole case, with
    just enough tags for :func:`miai_dicom.series.load_series` and
    SimpleITK's ``ImageSeriesReader`` to read it back as a real 3D
    volume (geometry + pixel data), not just parse individual files.
    """
    directory.mkdir(parents=True, exist_ok=True)
    series_instance_uid = generate_uid()
    study_instance_uid = generate_uid()
    depth, rows, columns = image.shape

    pixel_volume = np.clip(image, 0, 4095).astype(np.uint16)

    for slice_index in range(depth):
        sop_instance_uid = generate_uid()

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        dataset = FileDataset(
            filename_or_obj=None,  # type: ignore[arg-type]
            # pydicom's stub omits None even though the real
            # implementation accepts it for in-memory-only datasets.
            dataset=Dataset(),
            file_meta=file_meta,
            preamble=b"\x00" * 128,
        )
        dataset.PatientID = f"EXAMPLE{case_index:03d}"
        dataset.PatientName = f"Example^Case{case_index:03d}"
        dataset.StudyInstanceUID = study_instance_uid
        dataset.SeriesInstanceUID = series_instance_uid
        dataset.SOPInstanceUID = sop_instance_uid
        dataset.Modality = "CT"
        dataset.Rows = rows
        dataset.Columns = columns
        dataset.InstanceNumber = slice_index + 1
        dataset.ImagePositionPatient = [0.0, 0.0, float(slice_index)]
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.SliceThickness = 1.0
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 12
        dataset.HighBit = 11
        dataset.PixelRepresentation = 0
        dataset.RescaleIntercept = 0
        dataset.RescaleSlope = 1
        dataset.PixelData = pixel_volume[slice_index].tobytes()

        dataset.save_as(str(directory / f"slice_{slice_index:03d}.dcm"))


def _write_label(path: Path, label: npt.NDArray[np.uint8]) -> None:
    """Write ``label`` as a NIfTI volume matching the DICOM series' geometry.

    Segmentation labels don't exist in DICOM itself, so -- unlike the
    image -- this goes straight to NIfTI via SimpleITK, the same
    library every other MIAI package uses for image I/O.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    label_image = sitk.GetImageFromArray(label)
    label_image.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(label_image, str(path))


def generate_synthetic_dataset() -> list[str]:
    """Generate ``NUM_CASES`` synthetic DICOM series + matching NIfTI labels.

    Returns:
        The label file path for each case, in the same order
        :class:`~miai_pipeline.stages.dicom_to_nifti.DicomToNiftiStage`
        will discover the matching DICOM series in --
        ``miai_dicom.series.load_series`` walks ``DICOM_DIR`` with
        ``sorted(directory.rglob("*"))``, so naming case directories
        ``case_000``, ``case_001``, ... guarantees this order matches
        without needing to inspect series UIDs after the fact.
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    rng = np.random.default_rng(0)
    label_paths = []
    for case_index in range(NUM_CASES):
        image, label = _make_labeled_volume(rng)

        case_dicom_dir = DICOM_DIR / f"case_{case_index:03d}"
        _write_dicom_series(case_dicom_dir, image, case_index)

        label_path = LABEL_DIR / f"case_{case_index:03d}_label.nii.gz"
        _write_label(label_path, label)
        label_paths.append(str(label_path))

    logger.info("Generated %d synthetic case(s) under %s", NUM_CASES, OUTPUT_DIR)
    return label_paths


def main() -> None:
    configure_logging(level="INFO", force=True)

    label_paths = generate_synthetic_dataset()

    config = PipelineConfig.from_yaml(CONFIG_PATH)
    pipeline = Pipeline.from_config(config)

    context = PipelineContext()
    context.set("dicom_dir", DICOM_DIR)
    context.set("label_paths", label_paths)

    result = pipeline.run(context)

    manifest = result.require("manifest")
    checkpoint_path = result.require("model_checkpoint_path")
    prediction_paths = result.require("prediction_paths")
    metrics = result.require("metrics")

    print()
    print("=== Pipeline finished ===")
    print(
        f"Dataset split: {len(manifest['train'])} train, "
        f"{len(manifest['val'])} val, {len(manifest['test'])} test"
    )
    print(f"Model checkpoint: {checkpoint_path}")
    print(f"Predictions written: {len(prediction_paths)}")
    print(f"Mean metrics: {metrics['mean']}")
    print(
        f"Full evaluation report: {CONFIG_PATH.parent.parent / 'output' / 'evaluation_report.json'}"
    )


if __name__ == "__main__":
    main()
