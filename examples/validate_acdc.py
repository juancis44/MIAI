"""Real-data validation: multi-class (RV/myocardium/LV) segmentation on ACDC.

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

**Binary, not multi-class -- iterations 1 through 4.** ACDC's ground
truth has 4 classes (background, right ventricle, myocardium, left
ventricle), but through the fourth iteration :mod:`miai_segmentation`
was binary-only (sigmoid + 0.5 threshold, ``DiceLoss(sigmoid=True)``),
so this script merged the three foreground structures into one
"whole heart" label rather than extending the core library --
deliberately scoped out as "a real feature addition, not a validation
task" at the time. See ``docs/real_data_validation.md`` for that
history and the **Fifth iteration** section below for where it stops
being deliberately out of scope.

**Second iteration: every improvement lever pulled at once.** The
first pass (30 patients, ED-frame-only, a small 2-level UNet, one
random flip, 40 epochs) overfit badly (val Dice 0.83, test Dice
0.088). This version scales up every lever simultaneously rather than
one at a time: more data (50 patients x ED+ES = up to 100 cases, a
patient-level split so a patient's ED and ES frames always land in
the same split -- otherwise the same anatomy would leak across
train/test), stronger augmentation (rotation and intensity shift in
addition to the flip), a deeper architecture (3 downsamples instead
of 2), finer resampling spacing, and more epochs.

**No resampled labels through the pipeline's own preprocessing
stage.** :class:`~miai_pipeline.stages.preprocessing.PreprocessingStage`
only ever resamples one list of volumes at a time, and MIAI's
pipeline conventionally leaves label volumes untouched by it (see
``examples/segmentation_pipeline.py``), which only works there because
the synthetic labels are already at the target spacing. Real ACDC
labels are not, so this script runs the *same* stage twice -- once
over the images (linear interpolation + z-score normalization), once
over the (uint8-cast, since the fifth iteration) labels
(nearest-neighbor, no normalization) --
so both land on identical geometry without touching
:mod:`miai_pipeline` itself.

**Patient-level split, not :class:`~miai_pipeline.stages.dataset.
DatasetStage`.** That stage shuffles and splits at the case level,
which is fine when each patient contributes exactly one case (the
first iteration's ED-only scope) but would let a patient's ED and ES
frames -- the same anatomy -- land in different splits here. This
script builds the manifest itself: patients (not cases) are shuffled
and partitioned by the same fractions, then every case belonging to a
chosen patient goes to that patient's split.

**Third iteration: 2D per-slice, not 3D.** ACDC's cine-MRI is acquired
as a stack of independent 2D short-axis slices (each its own breath-hold
acquisition), not a true volumetric scan -- in-plane resolution is
~1.5-2mm but through-plane spacing is ~6-10mm with only 6-15 slices per
volume. The first two iterations ran a 3D UNet over this anyway,
imposing a spatial relationship between slices the acquisition never
actually has, and treating each ED/ES frame as a single training
example (at most ~100 volumes) rather than each *slice* as one. This
iteration switches ``architecture.modality`` to ``"two_d"``: MIAI's
existing per-slice UNet, wired into every pipeline stage since Phase 8
(``expand_to_slice_dicts`` turns each ED/ES volume into one training
example per slice at train time; :func:`~miai_segmentation.two_d.infer
.run_case_inference` reassembles slice predictions back into one
volume per case at inference time, so :class:`~miai_pipeline.stages.
evaluation.EvaluationStage` still scores whole cases against ground
truth unchanged). Both the slice axis (Z, ~6-15 per case) and the time
axis (the two annotated cardiac phases, ED and ES, already loaded per
patient since the second iteration) now multiply out into independent
2D training examples -- roughly 60 volumes x ~12 slices ~= 700+
examples, versus 60 in the 3D runs, without staging a single extra
file. See ``docs/real_data_validation.md`` for the result.

**Fourth iteration: the full 150-patient dataset, not a 50-patient
subset.** Now that the 2D per-slice modality is confirmed to
generalize (iteration 3), ``DEFAULT_PATIENTS`` scales up from every
3rd patient to every patient, patient001 through patient150 -- 300
cases (both ED and ES), roughly 3x the training data of iteration 3.
Same architecture, patient-level split, augmentation, and epoch budget
otherwise. See ``docs/real_data_validation.md`` for the result.

**Fifth iteration: multi-class (RV/myocardium/LV), a real feature
addition to** :mod:`miai_segmentation`. Every prior iteration merged
ACDC's three annotated structures into one "whole heart" foreground
label -- useful to validate the pipeline and the 2D-per-slice
modeling choice cheaply, but clinically the three structures matter
individually (RV and LV volumes/ejection fractions are diagnostic
quantities in their own right; the myocardium is a distinct tissue
with its own pathology). This iteration extends
:mod:`miai_segmentation` and :mod:`miai_evaluation` with a genuine
multi-class path (see ``TrainingConfig.num_classes``,
``InferenceConfig.num_classes``, and ``MetricsConfig.num_classes``,
all newly added -- ``num_classes=1``, the default everywhere, keeps
every prior binary behavior byte-for-byte unchanged): softmax logits
and ``DiceLoss(softmax=True, to_onehot_y=True)`` instead of
sigmoid/threshold, argmax instead of a probability threshold at
inference, and one-hot-encoded, background-excluded (``include_
background=False``) evaluation metrics -- plus a per-class Dice
breakdown (``dice_class_1``/``dice_class_2``/``dice_class_3``, named
here as RV/Myo/LV via ``_CLASS_NAMES``, ACDC's own convention) so a
single macro-averaged number can't hide which structure the model
struggles with. No new data, no new staging, and no binarization step
any more -- ``_prepare_label`` (formerly ``_binarize_label``) now just
casts ACDC's already-4-class ground truth to ``uint8``, unchanged
otherwise from the fourth iteration's full 150-patient/300-case
dataset, split, augmentation, and epoch budget. See
``docs/real_data_validation.md`` for the result.

Run:
    python examples/validate_acdc.py --data-dir /path/to/ACDC \\
        --output-dir examples/output/acdc_validation
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import SimpleITK as sitk

from miai_core.io import write_json
from miai_core.logging import configure_logging, get_logger
from miai_evaluation.metrics import MetricsConfig
from miai_pipeline.context import PipelineContext
from miai_pipeline.stages.evaluation import EvaluationStage, EvaluationStageConfig
from miai_pipeline.stages.inference import InferenceStage, InferenceStageConfig
from miai_pipeline.stages.preprocessing import PreprocessingConfig, PreprocessingStage
from miai_pipeline.stages.training import TrainingStage, TrainingStageConfig
from miai_segmentation.modality import SegmentationInferenceConfig, SegmentationModalityConfig
from miai_segmentation.three_d.train import TrainingConfig
from miai_segmentation.two_d.infer import InferenceConfig
from miai_segmentation.two_d.models import ArchitectureConfig, UNetConfig
from miai_transforms.config import TransformConfig, TransformSpec

logger = get_logger(__name__)

#: The full ACDC dataset: patient001 through patient150 (up from the
#: second/third iterations' 50-patient subset), covering both the
#: official training split (001-100, 5 pathology groups of 20) and
#: testing split (101-150). Each patient contributes both its ED and
#: ES frames (see ``_all_frame_paths``), so this is 300 cases, not 150
#: -- the fourth iteration's scale-up now that the 2D per-slice
#: modality (third iteration) proved the architecture generalizes;
#: see ``docs/real_data_validation.md``.
DEFAULT_PATIENTS = [f"patient{i:03d}" for i in range(1, 151)]

#: Finer than the first iteration's (2.5, 2.5, 8.0) -- more spatial
#: detail survives resampling, at the cost of more voxels per case.
_TARGET_SPACING = (2.0, 2.0, 6.0)

#: Matches _ARCHITECTURE's three stride-2 downsamples (2 * 2 * 2 = 8) --
#: real volumes resampled to a fixed physical spacing (rather than a
#: fixed voxel grid, unlike the synthetic examples/tests) land on an
#: arbitrary size per case, which breaks the UNet's skip connections
#: unless every case is padded up to a multiple of this first (in-plane
#: (X, Y) is what matters for the 2D per-slice network this iteration
#: uses; padding Z too is harmless -- it just adds a few background-only
#: slices). Done once on disk (see ``_pad_to_divisible``) rather than
#: via a ``"divisible_pad"`` transform in train/val/test transforms: a
#: transform-only pad would apply to what the model sees during
#: inference but not to the *unpadded* preprocessed label
#: :class:`~miai_pipeline.stages.evaluation.EvaluationStage` reads
#: straight off disk, causing the same prediction/ground-truth
#: shape mismatch worked around in ``_resample_labels_to_reference``.
_DIVISIBLE_K = 8

#: Random rotation and intensity shift added on top of the second
#: iteration's random flip -- more varied augmentation to fight the
#: small-sample overfitting the first two iterations showed directly.
#: ``extract_slice`` (:class:`~miai_transforms.slice_transforms
#: .ExtractSliced`) runs right after loading, before any augmentation,
#: so ``rand_flip``/``rand_rotate90`` operate on the already-2D
#: ``(C, H, W)`` slice -- their axis-0/1 arguments mean image rows/
#: columns here, not depth (as they would on the whole-volume ``(C, D,
#: H, W)`` array the first two, 3D-modality iterations flipped/rotated
#: instead).
_TRAIN_TRANSFORMS = TransformConfig(
    transforms=[
        TransformSpec(name="load_image", params={"keys": ["image", "label"]}),
        TransformSpec(name="extract_slice", params={"keys": ["image", "label"]}),
        TransformSpec(
            name="rand_flip", params={"keys": ["image", "label"], "prob": 0.5, "spatial_axis": 0}
        ),
        TransformSpec(
            name="rand_rotate90",
            params={"keys": ["image", "label"], "prob": 0.5, "spatial_axes": (0, 1)},
        ),
        TransformSpec(
            name="rand_shift_intensity", params={"keys": ["image"], "prob": 0.5, "offsets": 0.1}
        ),
        TransformSpec(name="ensure_type", params={"keys": ["image", "label"]}),
    ]
)
_EVAL_TRANSFORMS = TransformConfig(
    transforms=[
        TransformSpec(name="load_image", params={"keys": ["image", "label"]}),
        TransformSpec(name="extract_slice", params={"keys": ["image", "label"]}),
        TransformSpec(name="ensure_type", params={"keys": ["image", "label"]}),
    ]
)
_TEST_TRANSFORMS = TransformConfig(
    transforms=[
        TransformSpec(name="load_image", params={"keys": ["image"]}),
        TransformSpec(name="extract_slice", params={"keys": ["image"]}),
        TransformSpec(name="ensure_type", params={"keys": ["image"]}),
    ]
)

#: Number of segmentation classes, including background -- ACDC's own
#: ground-truth convention: 0 = background, 1 = right ventricle (RV),
#: 2 = myocardium (Myo), 3 = left ventricle (LV). See the module
#: docstring's "Fifth iteration" section.
_NUM_CLASSES = 4

#: Human-readable names for :func:`miai_evaluation.metrics.
#: compute_case_metrics`'s generic ``dice_class_{c}`` keys -- kept here,
#: not in :mod:`miai_evaluation`, since that module stays
#: dataset-agnostic on purpose (see ``MetricsConfig.num_classes``'s
#: docstring) and this mapping is ACDC-specific domain knowledge.
_CLASS_NAMES = {1: "RV", 2: "Myo", 3: "LV"}

#: 2D per-slice UNet (see the module docstring's "Third iteration"
#: section for why 2D, not 3D, is the right fit for this data): a
#: third stride-2 level (16->32->64->128 channels) and 2 residual units
#: per level, the same depth/width as the second iteration's 3D
#: network, just at ``spatial_dims=2``. ``out_channels=_NUM_CLASSES``
#: (up from the binary iterations' implicit ``1``) is what actually
#: makes this a multi-class model -- see the module docstring's "Fifth
#: iteration" section for how ``TrainingConfig``/``InferenceConfig``/
#: ``MetricsConfig`` pick up the same ``_NUM_CLASSES`` to train, infer,
#: and score consistently as 4-class instead of binary.
_ARCHITECTURE = SegmentationModalityConfig(
    modality="two_d",
    two_d=ArchitectureConfig(
        kind="unet",
        unet=UNetConfig(
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            out_channels=_NUM_CLASSES,
        ),
    ),
)

#: 2D sliding-window ROI (in-plane only -- the 2D modality's inference
#: reassembles predictions slice by slice, see
#: :func:`~miai_segmentation.two_d.infer.run_case_inference`). Large
#: enough to cover every padded case's (X, Y) extent in one window (the
#: largest padded case in this dataset/spacing combination is
#: (216, 240, ...); this is comfortably above that, and a multiple of 8
#: to match ``_DIVISIBLE_K``). This isn't just a tuning choice -- it's
#: the same correctness requirement the second, 3D-modality iteration
#: found the hard way (see ``docs/real_data_validation.md``): the
#: per-slice validation loop scores each slice with a single
#: full-slice forward pass (no windowing at all, since
#: :class:`~miai_transforms.slice_transforms.ExtractSliced` already
#: reduced each item to one full 2D slice before it ever reaches the
#: model), so :class:`~miai_pipeline.stages.inference.InferenceStage`'s
#: sliding-window inference must see the same full slice per window
#: too, or test-time predictions come from a systematically different
#: computation than what validation measured.
_INFERENCE_ROI_SIZE = (256, 256)


def _all_frame_paths(data_dir: Path, patient: str) -> list[tuple[Path, Path]]:
    """Find every annotated (image, ground-truth) NIfTI pair for a patient.

    ACDC annotates two cardiac phases per patient -- end-diastole (ED)
    and end-systole (ES) -- and its per-patient filenames encode the
    actual acquisition frame number (e.g. ``patient001_frame01.nii.gz``,
    ``patient001_frame12.nii.gz``), which varies patient to patient.
    Returns both pairs, sorted by frame number (ED first).
    """
    patient_dir = data_dir / patient
    candidates = sorted(
        p for p in patient_dir.glob(f"{patient}_frame*.nii.gz") if not p.name.endswith("_gt.nii.gz")
    )
    if not candidates:
        raise FileNotFoundError(f"No frame*.nii.gz files found under {patient_dir}")
    pairs = []
    for image_path in candidates:
        label_path = image_path.with_name(image_path.name.replace(".nii.gz", "_gt.nii.gz"))
        if not label_path.exists():
            raise FileNotFoundError(f"Expected ground truth at {label_path}, not found.")
        pairs.append((image_path, label_path))
    return pairs


def _prepare_label(src: Path, dst: Path) -> None:
    """Copy a ground-truth label, cast to ``uint8``.

    Fifth iteration: no longer binarized (see the module docstring's
    "Fifth iteration" section) -- ACDC's ground truth already encodes
    exactly the four classes this iteration trains on (background=0,
    RV=1, myocardium=2, LV=3), so the only transformation needed is a
    consistent dtype, matching what every downstream step (resampling,
    padding, one-hot encoding in :func:`miai_evaluation.metrics.
    compute_case_metrics`) assumes.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    label_image = sitk.ReadImage(str(src))
    cast = sitk.Cast(label_image, sitk.sitkUInt8)
    cast.CopyInformation(label_image)
    sitk.WriteImage(cast, str(dst))


def build_case_lists(
    data_dir: Path, patients: list[str], output_dir: Path
) -> tuple[list[Path], list[Path], list[str]]:
    """Discover every ED+ES frame for each patient and prepare their labels.

    Returns:
        Parallel ``(image_paths, label_paths, patient_ids)`` lists, one
        entry per (patient, frame) case -- ``patient_ids`` records
        which patient each case came from, for the patient-level split
        in :func:`_patient_level_split`.
    """
    image_paths = []
    prepared_label_paths = []
    patient_ids = []
    for patient in patients:
        for image_path, label_path in _all_frame_paths(data_dir, patient):
            case_name = image_path.name.removesuffix(".nii.gz")
            prepared_label_path = output_dir / "labels" / f"{case_name}_gt.nii.gz"
            _prepare_label(label_path, prepared_label_path)
            image_paths.append(image_path)
            prepared_label_paths.append(prepared_label_path)
            patient_ids.append(patient)
    return image_paths, prepared_label_paths, patient_ids


def _patient_level_split(
    image_paths: list[Path],
    label_paths: list[Path],
    patient_ids: list[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    manifest_path: Path,
) -> dict[str, list[object]]:
    """Split cases into train/val/test by patient, not by case.

    Splitting at the case level (as
    :class:`~miai_pipeline.stages.dataset.DatasetStage` does) would let
    the same patient's ED and ES frames -- the same anatomy -- land in
    different splits, leaking information from train into test.
    Instead, whole patients are shuffled and partitioned by the given
    fractions, and every case belonging to a chosen patient goes to
    that patient's split.
    """
    unique_patients = sorted(set(patient_ids))
    random.Random(seed).shuffle(unique_patients)

    n = len(unique_patients)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_patients = set(unique_patients[:n_test])
    val_patients = set(unique_patients[n_test : n_test + n_val])
    train_patients = set(unique_patients[n_test + n_val :])

    def _entries(patients: set[str]) -> list[object]:
        return [
            {"image": str(image_paths[i]), "label": str(label_paths[i])}
            for i in range(len(patient_ids))
            if patient_ids[i] in patients
        ]

    manifest: dict[str, list[object]] = {
        "train": _entries(train_patients),
        "val": _entries(val_patients),
        "test": _entries(test_patients),
    }
    logger.info(
        "Patient-level split: %d train patients (%d cases), %d val patients (%d cases), "
        "%d test patients (%d cases)",
        len(train_patients),
        len(manifest["train"]),
        len(val_patients),
        len(manifest["val"]),
        len(test_patients),
        len(manifest["test"]),
    )
    write_json(manifest, str(manifest_path))
    return manifest


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

    image_paths, label_paths, patient_ids = build_case_lists(data_dir, DEFAULT_PATIENTS, output_dir)
    logger.info(
        "Discovered %d ACDC cases (ED+ES) across %d patients under %s",
        len(image_paths),
        len(set(patient_ids)),
        data_dir,
    )

    context = PipelineContext()

    # Preprocess images and (prepared, multi-class) labels through the
    # *same* stage separately, so both land on identical resampled
    # geometry -- see
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

    manifest = _patient_level_split(
        padded_image_paths,
        padded_label_paths,
        patient_ids,
        val_fraction=0.2,
        test_fraction=0.2,
        seed=42,
        manifest_path=output_dir / "manifest.json",
    )
    context.set("manifest", manifest)
    context.set("manifest_path", str(output_dir / "manifest.json"))

    training_stage = TrainingStage(
        TrainingStageConfig(
            checkpoint_dir=str(output_dir / "checkpoints"),
            train_transforms=_TRAIN_TRANSFORMS,
            val_transforms=_EVAL_TRANSFORMS,
            architecture=_ARCHITECTURE,
            training=TrainingConfig(
                max_epochs=max_epochs,
                learning_rate=1e-3,
                device="cpu",
                num_classes=_NUM_CLASSES,
            ),
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
                two_d=InferenceConfig(
                    roi_size=_INFERENCE_ROI_SIZE,
                    sw_batch_size=4,
                    overlap=0.25,
                    device="cpu",
                    num_classes=_NUM_CLASSES,
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
                num_classes=_NUM_CLASSES,
            ),
            report_path=str(output_dir / "evaluation_report.json"),
        )
    )
    context = evaluation_stage.run(context)
    metrics = context.require("metrics")

    # Human-readable RV/Myo/LV names for compute_case_metrics's generic
    # dice_class_{c} keys -- see _CLASS_NAMES for why this mapping lives
    # here, not in miai_evaluation.
    named_class_dice = {
        f"dice_{name.lower()}": metrics["mean"][f"dice_class_{class_id}"]
        for class_id, name in _CLASS_NAMES.items()
        if f"dice_class_{class_id}" in metrics["mean"]
    }
    if named_class_dice:
        logger.info("Per-class mean test Dice: %s", named_class_dice)

    return {
        "manifest_sizes": {k: len(v) for k, v in manifest.items()},
        "checkpoint": context.require("model_checkpoint_path"),
        "mean_metrics": metrics["mean"],
        "named_class_dice": named_class_dice,
        "per_case": metrics["per_case"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="ACDC root dir with patientXXX/ subfolders"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output/acdc_validation"))
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=25,
        help="25 is what the fourth iteration (full 150-patient dataset) used; the third "
        "iteration's smaller 50-patient subset used 40 -- pass explicitly to match either.",
    )
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
