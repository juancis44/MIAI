# Real-data validation: ACDC

Every test and example in this repo before 2026-08-26 ran against small
synthetic volumes -- useful for exercising the wiring, but it never proved
`miai_pipeline` behaves sensibly on real clinical anatomy, real intensity
distributions, or the messy per-patient variation real MRI actually has
(different scanner matrix sizes, slice counts, spacings). This page records
the first such validation, run against the public **ACDC** (Automated
Cardiac Diagnosis Challenge) cardiac cine-MRI dataset.

> O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al. "Deep Learning
> Techniques for Automatic MRI Cardiac Multi-structures Segmentation and
> Diagnosis: Is the Problem Solved?" in *IEEE Transactions on Medical
> Imaging*, vol. 37, no. 11, pp. 2514-2525, Nov. 2018.
> doi: [10.1109/TMI.2018.2837502](https://doi.org/10.1109/TMI.2018.2837502)

The runnable script is `examples/validate_acdc.py`. ACDC itself is **not**
bundled with MIAI -- it's ~2GB and its license requires accepting terms
before download (see the
[ACDC challenge page](https://www.creatis.insa-lyon.fr/Challenge/acdc/)).
Point `--data-dir` at your own copy to reproduce this.

## Scope, and why it was simplified

**Binary "whole heart" segmentation, not multi-class.** ACDC's ground truth
has 4 classes (background, right ventricle, myocardium, left ventricle),
but `miai_segmentation` is currently binary-only (sigmoid + 0.5 threshold,
`DiceLoss(sigmoid=True)`, `AsDiscrete(threshold=0.5)`). Extending the core
library to multi-class (softmax/argmax output, per-class Dice, a
multi-channel loss) would be a real feature addition, not a validation
task -- so this validation merges the three foreground structures into one
label instead. Multi-class support is a natural next step if per-structure
(RV/myocardium/LV) metrics are needed later.

**One frame (end-diastole) per patient.** ACDC annotates two cardiac
phases per patient, ED and ES. Using both would let the same patient's
anatomy leak across the train/val/test split, since
`DatasetStage` splits at the case level, not the patient level -- so each
patient contributes only its ED frame, keeping every split's cases from
distinct patients.

**30 patients, not all 150.** Every 5th patient ID from `patient001` to
`patient146` (`examples/validate_acdc.py`'s `DEFAULT_PATIENTS`), spread
across the full numeric range so both ACDC's official training split
(patient001-100, five 20-patient pathology groups) and its testing split
(patient101-150) are represented. Deterministic, not randomly sampled, so
re-running the script always validates against the same cases. This keeps
the run fast enough for a CPU-only sandbox (no GPU); scaling up to the
full 150 patients (or both ED+ES with a patient-level split) is
straightforward with more compute/time.

## Bugs found (and fixed) by running against real data

Real ACDC volumes broke two assumptions synthetic data never exercised
before:

1. **Independently resampling images and labels can silently disagree by
   a voxel.** `PreprocessingStage` only ever resamples one list of volumes
   at a time, and other MIAI examples get away with leaving label volumes
   untouched by it because their synthetic labels are already at the
   target spacing. Real ACDC labels aren't -- and running the same
   `target_spacing` through `PreprocessingStage` twice (once for images,
   once for labels) rounds each side's output size independently from
   `round(original_size * original_spacing / target_spacing)`. ACDC's own
   image/label NIfTI headers don't describe *exactly* identical geometry
   (SimpleITK's "unexpected scales in sform" warning on every ACDC file
   hints at this), so the two rounded to different sizes by 1 voxel and
   MONAI's `DiceLoss` raised a shape-mismatch `AssertionError`. Fixed in
   the validation script (not in `miai_pipeline` itself) by resampling
   each label directly onto its own already-preprocessed image as the
   SimpleITK reference geometry, which can't disagree by construction.
2. **A 3D UNet needs its input padded to a multiple of its total
   downsampling stride, and nothing in MIAI enforced that.** Every
   synthetic test/example volume happens to already be sized that way by
   construction. Real volumes resampled to a fixed *physical* spacing
   (rather than a fixed voxel grid) land on an arbitrary size per case --
   training crashed with a skip-connection concatenation shape mismatch
   inside `monai.networks.nets.UNet`. Fixed two ways: the validation
   script now pads every (image, label) pair to a multiple of the
   architecture's stride product before they ever reach the pipeline
   (`_pad_to_divisible`, done once on disk so training, inference's
   reference geometry, and evaluation's ground truth all agree); and
   `miai_transforms.compose.TRANSFORM_REGISTRY` gained a general-purpose
   `"divisible_pad"` entry (`monai.transforms.DivisiblePadd`) so future
   pipelines needing this can reach it through a normal YAML transform
   spec instead of writing their own SimpleITK padding code, with a
   dedicated test (`tests/test_transforms_compose.py::
   test_divisible_pad_pads_to_a_multiple_of_k`).

Both are now documented in `examples/validate_acdc.py`'s module docstring
and inline comments, and the second is a small, permanent, tested addition
to the reusable transform registry -- not a one-off workaround.

## What the run showed

Config: `UNetConfig(channels=(16, 32, 64), strides=(2, 2), num_res_units=1)`,
`target_spacing=(2.5, 2.5, 8.0)` mm, Adam `lr=1e-3`, 40 epochs, CPU-only,
18 train / 6 val / 6 test cases (70/20/... split of the 30-patient subset,
`DatasetStage(val_fraction=0.2, test_fraction=0.2, seed=42)`).

| Split | Cases | Dice |
|---|---|---|
| Validation (best epoch, 40) | 6 | 0.83 |
| Held-out test | 6 | 0.088 |

The full per-case test-set report, `evaluation_report.json`, also includes
Hausdorff distance, IoU, sensitivity, specificity, and volume similarity
per case.

**The pipeline itself is validated: it ran the full DICOM-free path
(preprocess -> split -> train -> sliding-window inference -> evaluate)
against real clinical MRI end to end, with no crashes once the two bugs
above were fixed, and produced metrics in the expected shape and range.**

**The trained model is not clinically useful, and that's expected.** Val
Dice climbing to 0.83 while test Dice collapses to 0.088 is classic
small-sample overfitting, confirmed by inspecting predictions directly: on
one held-out case the model marks **49.6%** of voxels as heart against a
true foreground of **5.6%** (`specificity` across the test set averages
0.47 -- barely better than a coin flip on background voxels). With only 18
training volumes and no augmentation beyond a single random flip, the
model learned to predict "about half the volume" rather than the heart's
actual shape. A model worth deploying would need substantially more
training data (the full 150-patient dataset, both ED and ES frames with a
patient-level split), more epochs with early stopping, stronger
augmentation, and likely multi-class labels for clinically meaningful
per-structure metrics -- none of which this validation's scope covered
by design (see "Scope, and why it was simplified" above).

## Reproducing this

```bash
python examples/validate_acdc.py \
    --data-dir /path/to/ACDC \
    --output-dir examples/output/acdc_validation \
    --max-epochs 40
```

Outputs land under `--output-dir` (git-ignored, like every other
`examples/output/` run): binarized labels, preprocessed/padded
images and labels, the manifest split, the checkpoint, per-case
predictions, and `evaluation_report.json`.
