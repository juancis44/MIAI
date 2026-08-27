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

## Second iteration: every improvement lever pulled at once

The first run's diagnosis pointed at several possible fixes: more
data, stronger augmentation, more epochs, a bigger architecture, finer
resampling spacing, or multi-class labels. Rather than trying them one
at a time, this iteration pulled every lever except multi-class labels
simultaneously, to see how much combined headroom there was:

| Lever | First run | Second run |
|---|---|---|
| Patients | 30, ED only | 50, ED **and** ES (up to 100 cases) |
| Split | Case-level (`DatasetStage`) | **Patient-level** (a patient's ED/ES frames always share a split) |
| Augmentation | Random flip only | Flip + random 90 deg rotation + random intensity shift |
| Architecture | 2-level UNet (16, 32, 64 ch), 1 res unit | 3-level UNet (16, 32, 64, 128 ch), 2 res units |
| Resampling spacing | (2.5, 2.5, 8.0) mm | (2.0, 2.0, 6.0) mm |
| Epochs | 40 | 60 |

Config: Adam `lr=1e-3`, CPU-only, `DEFAULT_PATIENTS` (every 3rd patient
from `patient001` to `patient148`), 60 train / 20 val / 20 test cases
(a 60/20/20 patient-level split of the 50-patient x ED+ES set).

### A real bug found along the way, not just overfitting

The first attempt at this run showed a mean test Dice of **0.040** --
worse than the first iteration, despite validation Dice reaching 0.74.
That gap was too large to be overfitting alone. The cause: finer
z-spacing and a deeper network left padded volumes up to 24 slices
tall, but `InferenceStage`'s sliding-window `roi_size` was still
`(96, 96, 8)` -- sized for the first iteration's shallower cases.
`train_model`'s validation loop scores each case with a single
full-volume forward pass (see `miai_segmentation/three_d/train.py`),
so windowing test-time inference into 8-slice chunks starved the model
of z-context it had during validation, producing predictions from a
systematically different computation than what validation measured
(mean sensitivity was just 0.14 -- the model barely predicted any
foreground at all under windowing). Fixed by sizing `roi_size` to
`(256, 256, 32)`, comfortably larger than any padded case, so sliding-
window inference reduces to a single full-volume pass matching
validation. Re-running evaluation from the same checkpoint with the
fix applied roughly doubled mean test Dice, from 0.040 to 0.084.

### What the (bug-fixed) run showed

| Split | Cases | Dice |
|---|---|---|
| Validation (best epoch, 60) | 20 | 0.72 |
| Held-out test | 20 | 0.082 |

**Combining every lever did not meaningfully improve real
generalization.** Test Dice (0.082) is statistically indistinguishable
from the first iteration's 0.088 -- despite 3.3x the cases, a
patient-level split closing the leakage risk, stronger augmentation, a
deeper network, and finer resampling. Per-case test Dice ranges from
0.036 to 0.148, and mean specificity (0.60) is barely better than
chance -- the model is still not learning heart shape from 60 training
cases. The honest read: at this scale, model capacity was not the
bottleneck (the same architecture reached 0.72 val Dice), *training
data volume relative to the model's capacity* still is. A materially
better result would likely need the full 150-patient dataset (not a
50-patient subset), explicit regularization (dropout, weight decay --
neither is currently exposed by `TrainingConfig`), and/or multi-class
labels giving the loss more structured signal per voxel -- none of
which this iteration covered.

## Third iteration: 2D per-slice, not 3D

The first two iterations both ran a 3D UNet, and both landed around
test Dice 0.08-0.09 no matter what else changed. The reason turned out
to be a modeling choice, not a data or training problem: ACDC's
cine-MRI is acquired as a stack of independent 2D short-axis slices
(each its own breath-hold acquisition), not a true volumetric scan --
in-plane resolution is ~1.5-2mm but through-plane spacing is ~6-10mm
with only 6-15 slices per case. Feeding that into a 3D UNet imposes a
spatial relationship between slices the acquisition never actually
has, and treats each ED/ES frame as one training example (at most ~100
volumes across the whole run) rather than each *slice* as one.

This iteration switches `architecture.modality` to `"two_d"` --
MIAI's per-slice UNet, wired into every pipeline stage since Phase 8.
No new data, no new staging: `expand_to_slice_dicts` turns each ED/ES
volume already on disk into one training example per slice, and
`miai_segmentation.two_d.infer.run_case_inference` reassembles slice
predictions back into one volume per case at inference time, so
evaluation still scores whole cases against ground truth, unchanged.
Both axes the user pointed at -- slices (Z) and time (the two
annotated cardiac phases, ED and ES) -- now multiply out into
independent 2D training examples: the same 60 training volumes become
roughly 700+ 2D slices per epoch.

Config: same 3-level UNet (16, 32, 64, 128 channels, 2 res units) at
`spatial_dims=2`, same patient-level 60/20/20 split, same augmentation
(flip + rotate90 + intensity shift, now applied to the already-2D
slice), Adam `lr=1e-3`, 40 epochs, CPU-only. Inference: a 2D
sliding-window `roi_size=(256, 256)`, large enough to cover any slice
in one window -- the same full-frame-per-forward-pass correctness
requirement the second iteration's `roi_size` bug taught (see above),
just in 2D this time.

| Split | Cases | Dice |
|---|---|---|
| Validation (best epoch, 40) | 20 | 0.88 |
| Held-out test | 20 | **0.71** |

**This is the first iteration where the model actually generalizes.**
Test Dice jumped from ~0.08 (both 3D iterations) to **0.71** -- roughly
9x -- with the same data, the same patient-level split, and the same
epoch budget as the second iteration, changed only in how the volume
is fed to the network. Mean specificity is 0.994 (near-perfect
background rejection, unlike the 3D runs' near-chance ~0.5-0.6) and
mean sensitivity is 0.73. Per-case test Dice ranges 0.35-0.89 -- still
real variance case to case (patient088's two frames are the weak
spot, Dice ~0.36; patient076's are the strongest, ~0.80-0.89) -- but
every case beats the *best* single case from either 3D iteration.
The val/test gap (0.88 vs 0.71) is now in a normal, expected range for
this data scale, rather than the near-total collapse the 3D iterations
showed. **The practical lesson: matching the model's inductive bias to
how the data was actually acquired mattered far more here than any of
the second iteration's levers (more data, augmentation, a deeper
network, finer spacing) combined.**

## Fourth iteration: the full 150-patient dataset

The third iteration proved the architecture (2D per-slice) was the
right fix; this iteration asks whether the second iteration's other
lever -- more data -- still helps now that the inductive bias is
correctly matched. `DEFAULT_PATIENTS` scales from every 3rd patient
(50 patients, 100 cases) to every patient, `patient001` through
`patient150` (150 patients, 300 cases: both ED and ES for each,
covering ACDC's full official training split 001-100 across 5
pathology groups, plus the testing split 101-150). Same 2D per-slice
architecture, same patient-level split logic, same augmentation
(flip + rotate90 + intensity shift) as the third iteration. Epoch
budget was reduced from 40 to 25 given ~3x the per-epoch example
count (roughly 2,160 2D slices/epoch vs ~700), to keep wall-clock
reasonable on CPU-only compute (~326s/epoch, confirmed by a 2-epoch
calibration run before the full run).

Config: same 3-level UNet (16, 32, 64, 128 channels, 2 res units) at
`spatial_dims=2`, patient-level split (this time 180/60/60 cases from
90/30/30 patients), Adam `lr=1e-3`, 25 epochs, CPU-only, 2D
sliding-window `roi_size=(256, 256)`.

| Split | Cases | Dice |
|---|---|---|
| Validation (best epoch, 25) | 60 | 0.90 |
| Held-out test | 60 | **0.82** |

**More data helped further, on top of the architecture fix.** Test
Dice rose from 0.71 (50 patients) to **0.82** (150 patients) -- a real
and substantial jump, not noise -- with everything else about the
model and training procedure unchanged. Mean specificity is 0.996 and
mean sensitivity is 0.81, both improved over the third iteration.
Per-case test Dice (60 cases) has a median of 0.86 and a mean of 0.82
(stdev 0.13): 15 of 60 cases score >=0.90, most cases cluster in the
0.75-0.94 range, and a handful are notable outliers on the low end --
2 cases score below 0.5 (the weakest at 0.27), and 7 score below 0.7.
Those low-Dice cases are consistent with what's already known to be
hard for cardiac segmentation without multi-class labels (thin or
faint structures, atypical anatomy, or partial-volume slices at the
top/bottom of the stack) rather than a new problem introduced by
scaling up. **The honest read: unlike the second iteration (where
pulling every lever on top of a mismatched 3D architecture bought
nothing), pulling the same "more data" lever on top of the
*correctly* matched 2D architecture bought a further ~15% relative
improvement in test Dice.** The remaining gap to the 0.90 validation
Dice, and the small cluster of low-scoring outliers, are the natural
next targets -- multi-class labels (RV / myocardium / LV instead of
binary foreground) and explicit regularization remain the two levers
this session deliberately did not pull.

## Reproducing this

```bash
python examples/validate_acdc.py \
    --data-dir /path/to/ACDC \
    --output-dir examples/output/acdc_validation \
    --max-epochs 40
```

To reproduce the fourth iteration's full-150-patient result specifically:

```bash
python examples/validate_acdc.py \
    --data-dir /path/to/ACDC \
    --output-dir examples/output/acdc_validation_150 \
    --max-epochs 25
```

Outputs land under `--output-dir` (git-ignored, like every other
`examples/output/` run): binarized labels, preprocessed/padded
images and labels, the manifest split, the checkpoint, per-case
predictions, and `evaluation_report.json`.
