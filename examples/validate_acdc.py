"""Real-data validation: whole-heart binary segmentation on ACDC.

Every other example/test in this repo runs against small synthetic
volumes -- useful for exercising the wiring, but it never proves the
pipeline behaves sensibly on real clinical anatomy, real intensity
distributions, or the messy per-patient variation (different scanner
matrix sizes, slice counts, spacings) real MRI actually has. This
script is that missing check: it runs MIAI's existing
``miai_pipeline`` stages, unmodified, end to end against the public
ACDC (Automated Cardiac Diagnosis Challenge) cardiac cine-MRI dataset.

ACDC is not bundled with MIAI (large, and its license requires
accepting terms/citation before download -- see
https://www.creatis.insa-lyon.fr/Challenge/acdc/). Get it yourself and
point ``--data-dir`` at the folder containing ``patientXXX/``
subdirectories (each with ``patientXXX_frameNN.nii.gz`` +
``patientXXX_frameNN_gt.nii.gz`` pairs, the standard ACDC layout).

**Binary, not multi-class.** ACDC's ground truth has 4 classes
(background, right ventricle, myocardium, left ventricle), but
:mod:`miai_segmentation` is currently binary-only (sigmoid + 0.5
threshold, ``DiceLoss(sigmoid=True)``). This script merges the three
foreground structures into one "whole heart" label rather than
extending the core library -- multi-class support (softmax/argmax,
per-class Dice) would be a real feature addition, not a validation
task. See ``docs/real_data_validation.md`` for the full writeup.

**One frame per patient, no resampled labels through the pipeline's
own preprocessing stage.** Each patient contributes only its
end-diastole (ED) frame -- using both ED and ES would let the same
patient's anatomy leak across the train/val/test split, since
:class:`~miai_pipeline.stages.dataset.DatasetStage` splits at the case
level. :class:`~miai_pipeline.stages.preprocessing.PreprocessingStage`
only ever resamples one list of volumes at a time and MIAI's
pipeline conventionally leaves label volumes untouched by it (see
``examples/segmentation_pipeline.py``), which only works there because
the synthetic labels are already at the target spacing. Real ACDC
labels are not, so this script runs the *same* stage twice -- once
over the images (linear interpolation + z-score normalization), once
over the (binarized) labels (nearest-neighbor, no normalization) --
so both land on identical geometry without touching
:mod:`miai_pipeline` itself.

Run:
    python examples/validate_acdc.py --data-dir /path/to/ACDC \\
        --output-dir examples/output/acdc_validation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import SimpleITK as sitk

from miai_core.logging import configure_logging, get_logger
from miai_evaluation.metrics import MetricsConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.stages.dataset import DatasetConfig, DatasetStage
from miai_pipeline.stages.evaluation import EvaluationStage, EvaluationStageConfig
from miai_pipeline.stages.inference import InferenceStage, InferenceStageConfig
from miai_pipeline.stages.preprocessing import PreprocessingConfig, PreprocessingStage
from miai_pipeline.stages.training import TrainingStage, TrainingStageConfig
from miai_segmentation.modality import SegmentationInferenceConfig, SegmentationModalityConfig
from miai_segmentation.three_d.infer import InferenceConfig
from miai_segmentation.three_d.models import ArchitectureConfig, UNetConfig
from miai_segmentation.three_d.train import TrainingConfig
from miai_transforms.config import TransformConfig, TransformSpec

logger = get_logger(__name__)

#: Every 5th patient from patient001 to patient146 (30 patients total),
#: spread across the full numeric range so both the official ACDC
#: training split (001-100, 5 pathology groups of 20) and testing
#: split (101-150) are represented. Deterministic and reproducible --
#: not randomly sampled -- so re-running this script always validates
#: against the same cases.
DEFAULT_PATIENTS = [f"patient{i:03d}" for i in range(1, 147, 5)]

_TARGET_SPACING = (2.5, 2.5, 8.0)

#: Matches _ARCHITECTURE's two stride-2 downsamples (2 * 2 = 4) --
#: real volumes resampled to a fixed physical spacing (rather than a
#: fixed voxel grid, unlike the synthetic examples/tests) land on an
#: arbitrary size per case, which breaks the UNet's skip connections
#: unless every case is padded up to a multiple of this first. Done
#: once on disk (see ``_pad_to_divisible``) rather than via a
#: ``"divisible_pad"`` transform in train/val/test transforms: a
#: transform-only pad would apply to what the model sees during
#: inference but not to the *unpadded* preprocessed label
#: :class:`~miai_pipeline.stages.evaluation.EvaluationStage` reads
#: straight off disk, causing the same prediction/ground-truth
#: shape mismatch worked around in ``_resample_labels_to_reference``.
_DIVISIBLE_K = 4

_TRAIN_TRANSFORMS = TransformConfig(
    transforms=[
        TransformSpec(name="load_image", params={"keys": ["image", "label"]}),
        TransformSpec(
            name="rand_flip", params={"keys": ["image", "label"], "prob": 0.5, "spatial_axis": 0}
        ),
        TransformSpec(name="ensure_type", params={"keys": ["image", "label"]}),
    ]
)
_EVAL_TRANSFORMS = TransformConfig(
    transforms=[
        TransformSpec(name="load_image", params={"keys": ["image", "label"]}),
        TransformSpec(name="ensure_type", params={"keys": ["image", "label"]}),
    ]
)
_TEST_TRANSFORMS = TransformConfig(
    transforms=[
        TransformSpec(name="load_image", params={"keys": ["image"]}),
        TransformSpec(name="ensure_type", params={"keys": ["image"]}),
    ]
)

_ARCHITECTURE = SegmentationModalityConfig(
    modality="three_d",
    three_d=ArchitectureConfig(
        kind="unet",
        unet=UNetConfig(channels=(16, 32, 64), strides=(2, 2), num_res_units=1),
    ),
)


def _ed_frame_paths(data_dir: Path, patient: str) -> tuple[Path, Path]:
    """Find one patient's end-diastole (image, ground-truth) NIfTI pair.

    ACDC's per-patient filenames encode the actual acquisition frame
    number (e.g. ``patient001_frame01.nii.gz``), which varies patient
    to patient -- so this can't be hardcoded as ``frame01`` for every
    case. The lowest frame number is always end-diastole (ACDC's
    ``Info.cfg`` records ``ED`` as the earlier of the two annotated
    frames in every case in this dataset).
    """
    patient_dir = data_dir / patient
    candidates = sorted(
        p for p in patient_dir.glob(f"{patient}_frame*.nii.gz") if not p.name.endswith("_gt.nii.gz")
    )
    if not candidates:
        raise FileNotFoundError(f"No frame*.nii.gz files found under {patient_dir}")
    image_path = candidates[0]
    label_path = image_path.with_name(image_path.name.replace(".nii.gz", "_gt.nii.gz"))
    if not label_path.exists():
        raise FileNotFoundError(f"Expected ground truth at {label_path}, not found.")
    return image_path, label_path


def _binarize_label(src: Path, dst: Path) -> None:
    """Write a whole-heart binary mask (any of RV/myocardium/LV -> 1)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    label_image = sitk.ReadImage(str(src))
    binary = sitk.Cast(label_image > 0, sitk.sitkUInt8)
    binary.CopyInformation(label_image)
    sitk.WriteImage(binary, str(dst))


def build_case_lists(
    data_dir: Path, patients: list[str], output_dir: Path
) -> tuple[list[Path], list[Path]]:
    """Discover ED frames for each patient and binarize their labels.

    Returns:
        Parallel ``(image_paths, binarized_label_paths)`` lists, one
        entry per patient.
    """
    image_paths = []
    binary_label_paths = []
    for patient in patients:
        image_path, label_path = _ed_frame_paths(data_dir, patient)
        binary_label_path = output_dir / "binary_labels" / f"{patient}_gt_binary.nii.gz"
        _binarize_label(label_path, binary_label_path)
        image_paths.append(image_path)
        binary_label_paths.append(binary_label_path)
    return image_paths, binary_label_paths


def _resample_labels_to_reference(
    label_paths: list[Path], reference_image_paths: list[Path], out_dir: Path
) -> list[Path]:
    """Resample each label onto its own preprocessed image's exact grid.

    Running :class:`PreprocessingStage` independently over images and
    labels (even with the identical ``target_spacing``) can round to
    different output sizes by a voxel here and there when a label's
    NIfTI header doesn't describe *exactly* the same geometry as its
    image (ACDC's own files trigger SimpleITK's "unexpected scales in
    sform" warning, hinting at exactly this kind of header drift) --
    each recomputes its output size independently from
    ``round(original_size * original_spacing / target_spacing)``, and a
    tiny spacing difference is enough to round either side. Resampling
    directly onto the already-preprocessed image as the reference
    avoids the two ever disagreeing, guaranteed by construction rather
    than by both computing the same rounding twice.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for label_path, reference_path in zip(label_paths, reference_image_paths, strict=True):
        label_image = sitk.ReadImage(str(label_path))
        reference_image = sitk.ReadImage(str(reference_path))
        resampled = sitk.Resample(
            label_image,
            reference_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            label_image.GetPixelID(),
        )
        out_path = out_dir / f"{Path(label_path).stem.removesuffix('.nii')}_preprocessed.nii.gz"
        sitk.WriteImage(resampled, str(out_path))
        out_paths.append(out_path)
    return out_paths


def _pad_to_divisible(
    image_paths: list[Path],
    label_paths: list[Path],
    k: int,
    image_out_dir: Path,
    label_out_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """Zero-pad each (image, label) pair's spatial size up to a multiple of ``k``.

    Padded once here, together, straight on disk -- rather than only
    at the model's input via a transform -- so every stage downstream
    (training, inference's reference geometry, and evaluation's
    ground truth, which :class:`~miai_pipeline.stages.evaluation.
    EvaluationStage` reads directly from the manifest, bypassing the
    transform pipeline entirely) sees the same, already-consistent
    size. See ``_DIVISIBLE_K``'s comment for why a transform-only pad
    doesn't work here.
    """
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)
    out_images = []
    out_labels = []
    for image_path, label_path in zip(image_paths, label_paths, strict=True):
        image = sitk.ReadImage(str(image_path))
        label = sitk.ReadImage(str(label_path))
        pad_upper = [(-extent) % k for extent in image.GetSize()]

        image_padder = sitk.ConstantPadImageFilter()
        image_padder.SetPadUpperBound(pad_upper)
        image_padder.SetConstant(0.0)
        padded_image = image_padder.Execute(image)

        label_padder = sitk.ConstantPadImageFilter()
        label_padder.SetPadUpperBound(pad_upper)
        label_padder.SetConstant(0)
        padded_label = label_padder.Execute(label)

        out_image_path = image_out_dir / image_path.name
        out_label_path = label_out_dir / label_path.name
        sitk.WriteImage(padded_image, str(out_image_path))
        sitk.WriteImage(padded_label, str(out_label_path))
        out_images.append(out_image_path)
        out_labels.append(out_label_path)
    return out_images, out_labels


def run_validation(data_dir: Path, output_dir: Path, max_epochs: int) -> dict[str, object]:
    """Run the full preprocess -> split -> train -> infer -> evaluate pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, label_paths = build_case_lists(data_dir, DEFAULT_PATIENTS, output_dir)
    logger.info("Discovered %d ACDC ED-frame cases under %s", len(image_paths), data_dir)

    context = PipelineContext()

    # Preprocess images and (binarized) labels through the *same* stage
    # separately, so both land on identical resampled geometry -- see
    # the module docstring for why this can't just reuse
    # examples/segmentation_pipeline.py's single-pass pattern.
    image_stage = PreprocessingStage(
        PreprocessingConfig(
            output_dir=str(output_dir / "preprocessed_images"),
            target_spacing=_TARGET_SPACING,
            interpolation="linear",
            normalization="zscore",
        )
    )
    context.set("nifti_paths", image_paths)
    context = image_stage.run(context)
    context.set("preprocessed_image_paths", context.require("preprocessed_paths"))

    preprocessed_image_paths = context.require("preprocessed_image_paths")
    preprocessed_label_paths = _resample_labels_to_reference(
        label_paths, preprocessed_image_paths, output_dir / "preprocessed_labels"
    )

    padded_image_paths, padded_label_paths = _pad_to_divisible(
        preprocessed_image_paths,
        preprocessed_label_paths,
        _DIVISIBLE_K,
        output_dir / "padded_images",
        output_dir / "padded_labels",
    )
    context.set("preprocessed_image_paths", padded_image_paths)
    context.set("preprocessed_label_paths", padded_label_paths)

    dataset_stage = DatasetStage(
        DatasetConfig(
            manifest_path=str(output_dir / "manifest.json"),
            context_key="preprocessed_image_paths",
            label_context_key="preprocessed_label_paths",
            val_fraction=0.2,
            test_fraction=0.2,
            seed=42,
        )
    )
    context = dataset_stage.run(context)
    manifest = context.require("manifest")
    logger.info(
        "Split: %d train, %d val, %d test",
        len(manifest["train"]),
        len(manifest["val"]),
        len(manifest["test"]),
    )

    training_stage = TrainingStage(
        TrainingStageConfig(
            checkpoint_dir=str(output_dir / "checkpoints"),
            train_transforms=_TRAIN_TRANSFORMS,
            val_transforms=_EVAL_TRANSFORMS,
            architecture=_ARCHITECTURE,
            training=TrainingConfig(max_epochs=max_epochs, learning_rate=1e-3, device="cpu"),
        )
    )
    context = training_stage.run(context)
    logger.info("Checkpoint: %s", context.require("model_checkpoint_path"))

    inference_stage = InferenceStage(
        InferenceStageConfig(
            output_dir=str(output_dir / "predictions"),
            transforms=_TEST_TRANSFORMS,
            architecture=_ARCHITECTURE,
            inference=SegmentationInferenceConfig(
                three_d=InferenceConfig(
                    roi_size=(96, 96, 8), sw_batch_size=1, overlap=0.25, device="cpu"
                )
            ),
        )
    )
    context = inference_stage.run(context)

    evaluation_stage = EvaluationStage(
        EvaluationStageConfig(
            metrics=MetricsConfig(
                include_dice=True,
                include_hausdorff=True,
                include_iou=True,
                include_sensitivity=True,
                include_specificity=True,
                include_volume_similarity=True,
            ),
            report_path=str(output_dir / "evaluation_report.json"),
        )
    )
    context = evaluation_stage.run(context)
    metrics = context.require("metrics")

    return {
        "manifest_sizes": {k: len(v) for k, v in manifest.items()},
        "checkpoint": context.require("model_checkpoint_path"),
        "mean_metrics": metrics["mean"],
        "per_case": metrics["per_case"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="ACDC root dir with patientXXX/ subfolders"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output/acdc_validation"))
    parser.add_argument("--max-epochs", type=int, default=25)
    args = parser.parse_args()

    configure_logging(level="INFO", force=True)
    summary = run_validation(args.data_dir, args.output_dir, args.max_epochs)

    print()
    print("=== ACDC real-data validation finished ===")
    print(json.dumps(summary["manifest_sizes"], indent=2))
    print(f"Checkpoint: {summary['checkpoint']}")
    print("Mean metrics:")
    print(json.dumps(summary["mean_metrics"], indent=2))


if __name__ == "__main__":
    main()
