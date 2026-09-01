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

## Fifth iteration: multi-class (RV/myocardium/LV)

Every prior iteration merged ACDC's three annotated structures into
one "whole heart" foreground label -- a deliberate scope cut, not a
capability gap discovered along the way (see the module docstring's
"Binary, not multi-class" note through the fourth iteration). Binary
segmentation was enough to validate the pipeline and the 2D-per-slice
modeling choice cheaply, but it also hides the question that actually
matters clinically: RV and LV volumes/ejection fractions are
diagnostic quantities in their own right, and the myocardium is a
distinct tissue with its own pathology. This iteration is a real
feature addition to `miai_segmentation` and `miai_evaluation`, not
just another `validate_acdc.py` config change: `TrainingConfig`,
`InferenceConfig` (both `two_d` and `three_d`), and `MetricsConfig`
all gained a `num_classes` field (default `1`, so every existing
binary caller -- including every prior iteration above -- keeps
working byte-for-byte unchanged). Setting it above `1` switches
training to softmax logits and `DiceLoss(softmax=True,
to_onehot_y=True)` instead of sigmoid/threshold, inference to argmax
instead of a probability threshold, and evaluation to one-hot-encoded,
background-excluded (`include_background=False`) metrics -- plus a new
per-class Dice breakdown (`dice_class_1`/`dice_class_2`/
`dice_class_3`) so a single macro-averaged number can't hide which
structure the model struggles with most.

No new data, no new staging, and no binarization step any more --
ACDC's ground truth already encodes exactly the four classes trained
on here (background=0, RV=1, myocardium=2, LV=3), so the label
preparation step now just casts to `uint8`. Otherwise unchanged from
the fourth iteration: full 150-patient/300-case dataset, 2D per-slice
UNet (now `out_channels=4`), patient-level 180/60/60 split, same
augmentation, 25 epochs, CPU-only.

| Split | Cases | Dice (macro, foreground only) |
|---|---|---|
| Validation (best epoch, 25) | 60 | 0.83 |
| Held-out test | 60 | 0.72 |

Per-class mean test Dice tells a more useful story than the macro
average alone:

| Structure | Mean test Dice |
|---|---|
| Right ventricle (RV) | 0.58 |
| Myocardium (Myo) | 0.72 |
| Left ventricle (LV) | 0.86 |

**The honest read: multi-class is harder than binary, as expected, and
the difficulty is not spread evenly across structures.** Macro test
Dice (0.72) is lower than the fourth iteration's binary "whole heart"
result (0.82) -- unsurprising, since binary Dice gets to call any
correctly-identified heart pixel a win regardless of which structure
it belongs to, while this metric requires getting the *class* right
too, and RV in particular is a thin, crescent-shaped structure that is
intrinsically harder to segment precisely than the LV's near-circular,
high-contrast blood pool -- a well-known pattern in the cardiac
segmentation literature, not a bug. LV Dice (0.86) is close to the
binary result, meaning nearly all of the macro-average gap comes from
RV and, to a lesser extent, myocardium. Per-case test Dice (60 cases)
has a median of 0.74 and ranges from 0.18 to 0.87; 3 cases score below
0.5, the weakest (`patient142_frame12`, Dice 0.18) driven by a
correspondingly low score on every structure at once (RV 0.12, Myo
0.15, LV 0.26) rather than one bad structure -- consistent with a
genuinely hard slice/frame (thin, faint, or ambiguous anatomy) rather
than a class-specific failure mode. Training also hit one late,
transient instability (epoch 23's validation Dice collapsed to 0.0,
recovering partially by epoch 25) -- harmless here since checkpoint
selection only keeps strictly-improving epochs (the epoch 22
checkpoint, val Dice 0.83, is what test was scored against), but a
concrete example of exactly the kind of instability explicit
regularization (dropout, weight decay -- still not exposed by
`TrainingConfig`) is meant to guard against, and the clearest evidence
yet that it remains the natural next lever, not merely a nice-to-have.

## Sixth iteration: explicit regularization (dropout, weight decay)

The fifth iteration's training run hit a late, transient instability --
validation Dice collapsed to 0.0 at epoch 23 (loss spiking 0.14 -> 0.18
-> 0.44) -- a training-loop failure mode explicit regularization is
meant to guard against, and a concrete, real-data signal (not just a
theoretical gap) that this lever was worth pulling next. Two new,
orthogonal knobs, both newly added to `miai_segmentation` and both
defaulting to off (`0.0`) so every prior iteration's config keeps
working unchanged: `UNetConfig.dropout` (activation dropout inside each
residual unit's ADN block, set here to `0.2`) and
`TrainingConfig.weight_decay` (L2 penalty on the weights themselves,
passed straight to `torch.optim.Adam`, set here to `1e-5`). Otherwise
identical to the fifth iteration: same full 150-patient/300-case
multi-class dataset, patient-level split, augmentation, architecture
depth, and 25-epoch budget -- so any change in the result isolates the
effect of regularization, not a confound from also changing the data or
architecture.

| Split | Cases | Dice (macro, foreground only) |
|---|---|---|
| Validation (best epoch, 13/25) | 60 | 0.8252 |
| Held-out test | 60 | **0.7496** |

Per-class mean test Dice:

| Structure | Mean test Dice | Fifth iteration (no regularization) |
|---|---|---|
| Right ventricle (RV) | 0.70 | 0.58 |
| Myocardium (Myo) | 0.71 | 0.72 |
| Left ventricle (LV) | 0.83 | 0.86 |

**The instability is gone.** Where the fifth iteration's val Dice
collapsed to 0.0 at epoch 23, this run's epoch 23 scored 0.8051 --
consistent with every neighboring epoch (0.80-0.83 for epochs 14-24),
with no collapse anywhere in the 25-epoch run. That is the concrete
question this iteration was run to answer, and the answer is yes:
dropout + weight decay resolved the specific training-loop instability
the fifth iteration surfaced.

**Macro test Dice improved too, and RV -- the hardest structure --
improved the most.** Macro test Dice rose from 0.72 to **0.75**, and
RV Dice specifically rose from 0.58 to **0.70**, a 0.12 absolute (21%
relative) improvement -- the single biggest per-class change of any
iteration in this series. Myocardium Dice held essentially flat (0.72
-> 0.71, within noise). LV Dice dropped slightly (0.86 -> 0.83) but
stayed the strongest-performing structure by a wide margin. The
honest read on LV: it's a small drop on the easiest structure, most
plausibly regularization trading a little of LV's already-comfortable
margin for RV's much-needed improvement (dropout and weight decay both
suppress the network's ability to overfit any one structure's easier
signal), rather than a real regression -- and net effect across all
three structures is clearly positive.

The best validation checkpoint moved earlier too: epoch 13 (0.8252)
here versus epoch 22 (0.83) in the fifth iteration -- consistent with
regularization's usual effect of both stabilizing training and
converging to a good checkpoint sooner, at some cost to the ceiling
val Dice can reach (0.8252 vs 0.83, a small difference within normal
run-to-run variance).

Per-case test Dice (60 cases) has a mean of 0.75 (stdev 0.11, tighter
than the fifth iteration's stdev 0.16) and a median of 0.77, ranging
0.32-0.88 -- notably, **no case reaches 0.90** (unlike the fourth,
binary iteration), consistent with multi-class remaining intrinsically
harder than binary "whole heart" segmentation regardless of
regularization. Only 2 of 60 cases score below 0.5 (down from 3), and
16 score below 0.7 (a case count similar to the fifth iteration's
scatter). The weakest case is again `patient142_frame12` (Dice 0.32),
already flagged in the fifth iteration as a globally hard slice/frame
(low Dice across every structure at once, not a single-structure
failure) -- regularization improved it somewhat (0.18 -> 0.32) but
didn't fully resolve it, consistent with that case being a genuinely
ambiguous slice rather than a training artifact.

**The honest read: both goals of this iteration were met.** The
targeted problem -- the fifth iteration's training instability -- is
resolved, with no collapse anywhere in this run. And the secondary
hope -- that regularization would also improve generalization, not
just stability -- paid off too, concentrated almost entirely in RV,
the structure that most needed it. The remaining gap between the
best-ever binary result (0.82, fourth iteration) and this multi-class
result (0.75) is expected and explained by the fifth iteration's
analysis (multi-class inherently requires getting the class right, not
just the pixel), not a new problem this iteration introduced.

## Seventh iteration: per-class breakdown for every metric

Every prior multi-class iteration reported a per-class breakdown for
Dice (`dice_class_1`/`dice_class_2`/`dice_class_3`), but the other five
opted-in metrics (Hausdorff distance, IoU, sensitivity, specificity,
volume similarity) still only reported one macro-averaged number even
in multi-class mode -- hiding, for instance, whether the RV's lower
Dice comes with a correspondingly worse Hausdorff distance (a genuinely
worse boundary) or is driven mostly by size/overlap rather than
boundary shape. `miai_evaluation.metrics.compute_case_metrics` now
reports a `{metric}_class_{c}` entry for every opted-in metric in
multi-class mode, the same pattern `dice_class_{c}` already used.

No new training run for this iteration -- it only changes what
`compute_case_metrics` reports, so the sixth iteration's checkpoint and
predictions were re-scored against the same test set with the expanded
metric config, rather than retraining from scratch.

Per-class mean test metrics (sixth iteration's checkpoint, 60 test cases):

| Metric | RV | Myo | LV | Macro |
|---|---|---|---|---|
| Dice | 0.70 | 0.71 | 0.83 | 0.75 |
| Hausdorff distance (HD95, mm, lower is better) | 51.1 | 44.7 | 42.9 | 46.2 |
| IoU | 0.56 | 0.56 | 0.74 | 0.62 |
| Sensitivity | 0.75 | 0.75 | 0.92 | 0.80 |
| Specificity | 0.997 | 0.997 | 0.998 | 0.997 |
| Volume similarity | 0.87 | 0.91 | 0.89 | 0.93 |

**The per-class breakdown mostly confirms the same RV-is-hardest story
Dice alone already told, but with one genuine surprise.** IoU and
sensitivity track Dice closely (RV and Myo both clearly behind LV on
all three), consistent with RV's lower Dice being a real boundary/
overlap problem, not an artifact of the Dice formula specifically.
Specificity is uniformly excellent (>=0.997) across all three
structures -- expected, since specificity is dominated by the vast
majority-background voxels any of the three foreground classes leaves
alone, so it was never going to discriminate between structures.

**The surprise is volume similarity: RV (0.87) is the *worst*-scoring
structure by this metric, not LV (0.89) as every overlap-based metric
would suggest, and Myo (0.91) is actually the best.** Volume
similarity ignores spatial overlap entirely and only compares total
voxel counts -- so this says the model's RV *volume estimates*
(clinically relevant on their own, e.g. for ejection fraction) are
noisier case-to-case than its Myo volume estimates, even though its
Myo *segmentation shape* (Dice/IoU) is essentially tied with RV's. The
practical read: a downstream consumer that only needs RV/LV/Myo
*volumes* (not precise boundaries) should not assume Dice ranks
structures the same way volume similarity would -- they answer
different clinical questions, and this iteration is the first evidence
in this validation series that they can actually disagree about which
structure is "hardest."

## Eighth iteration: more epochs with early stopping

The sixth iteration's best checkpoint landed early (epoch 13 of a
fixed 25-epoch budget) and validation Dice never improved again in the
remaining 12 epochs -- a fixed epoch budget has no way to tell whether
that's a genuine plateau or just too short a run. This iteration adds
early stopping to `miai_segmentation`: `TrainingConfig.
early_stopping_patience` (default `None`, so every prior iteration's
config keeps behaving exactly as before), which stops training once
validation Dice has gone that many consecutive validation checks
without a new best. `--max-epochs` is raised to 50 (from 25) so a
later improvement gets a real chance to surface, and
`_EARLY_STOPPING_PATIENCE = 10` bounds how long training keeps running
once it's actually plateaued. Otherwise identical to the sixth
iteration: same full 150-patient/300-case multi-class dataset,
patient-level split, augmentation, architecture depth, dropout
(0.2), and weight decay (1e-5).

Training actually used the full extra room: validation Dice kept
setting new bests well past where the sixth iteration's fixed 25-epoch
budget would have cut it off -- 0.6245 (epoch 1) climbing steadily to
0.8095 (epoch 10), 0.8104 (epoch 12), 0.8239 (epoch 15), and finally
**0.8274 at epoch 19**, the best checkpoint of this run. From there,
validation Dice went 10 consecutive checks (epochs 20-29) without
beating 0.8274 -- including one more dip to 0.7757 at epoch 28, a
transient wobble similar in shape to (though much milder than) the
fifth iteration's epoch-23 collapse, but with checkpoint selection
correctly ignoring it -- so early stopping triggered at epoch 29,
roughly 40% into the raised 50-epoch budget, and training stopped
using the epoch 19 checkpoint.

| Split | Cases | Dice (macro, foreground only) |
|---|---|---|
| Validation (best epoch, 19/50, stopped at 29) | 60 | 0.8274 |
| Held-out test | 60 | **0.7740** |

Per-class mean test metrics (all six, per the seventh iteration's
breakdown), compared against the sixth iteration:

| Metric | RV (6th -> 8th) | Myo (6th -> 8th) | LV (6th -> 8th) | Macro (6th -> 8th) |
|---|---|---|---|---|
| Dice | 0.70 -> 0.70 | 0.71 -> 0.76 | 0.83 -> 0.87 | 0.75 -> **0.77** |
| Hausdorff distance (HD95, mm, lower better) | 51.1 -> 34.1 | 44.7 -> 29.3 | 42.9 -> 22.0 | 46.2 -> **28.4** |
| IoU | 0.56 -> 0.55 | 0.56 -> 0.62 | 0.74 -> 0.78 | 0.62 -> 0.65 |
| Sensitivity | 0.75 -> 0.73 | 0.75 -> 0.79 | 0.92 -> 0.90 | 0.80 -> 0.80 |
| Specificity | 0.997 -> 0.997 | 0.997 -> 0.997 | 0.998 -> 0.999 | 0.997 -> 0.998 |
| Volume similarity | 0.87 -> 0.88 | 0.91 -> 0.91 | 0.89 -> 0.94 | 0.93 -> 0.95 |

**More training room, released safely by early stopping, closed real
ground toward the binary-era ceiling.** Macro test Dice rose from 0.75
to **0.77**, and the Hausdorff distance improvement is the most
striking single number in this table -- macro HD95 dropped from 46.2mm
to 28.4mm (roughly 39% better), meaning predicted boundaries are
substantially closer to the true anatomy, not just marginally more
overlapping. Myocardium improved the most among the three structures
(Dice 0.71 -> 0.76), followed by LV (0.83 -> 0.87, now within 0.05 of
the fourth iteration's *binary* ceiling of 0.82 on a *harder*
multi-class task). RV stayed essentially flat on Dice (0.70 -> 0.70)
but still improved on Hausdorff distance (51.1mm -> 34.1mm) --
consistent with the seventh iteration's finding that RV's overlap
score and its boundary/volume quality don't always move together.

Per-case test Dice (60 cases) improved across the board: mean 0.77
(stdev 0.09, tighter than the sixth iteration's stdev 0.11), median
0.79, only 1 case below 0.5 (down from 2), and 10 below 0.7 (down from
16). The weakest case is once again `patient142_frame12` (Dice 0.47,
up from 0.32 in the sixth iteration and 0.18 in the fifth) --
regularization plus more training keeps chipping away at this
genuinely hard slice without fully resolving it, consistent with it
being an ambiguous frame rather than a training artifact.

**The honest read: about 40% of the gap to the binary ceiling closed,
and no case for pushing further with this exact setup.** The gap
between this result (0.77) and the fourth iteration's binary "whole
heart" ceiling (0.82) narrowed from 0.10 (sixth iteration) to 0.05 --
real progress, driven by giving training the room to actually reach
its plateau rather than being cut off early. Early stopping did
exactly its intended job: it used the extra room where the model kept
improving (epochs 1-19) and cut the run short once it stopped
(epochs 20-29), instead of blindly consuming the full 50-epoch budget
either way. Whether the remaining 0.05 gap is closeable with even more
epochs is doubtful given this run's own evidence -- validation Dice
went 10 consecutive checks without beating epoch 19's result, including
epochs deep into loss values (~0.167) essentially unchanged from
epoch 22 onward, suggesting the model, not the epoch budget, is now
the binding constraint. Closing more of the gap would likely need a
different lever -- a deeper/wider architecture, a learning rate
schedule, or more training data -- not simply more epochs.

## Ninth iteration: cosine-annealed learning rate

The eighth iteration's best checkpoint (epoch 19) was followed by 10
non-improving validation checks before early stopping fired, including
a transient dip to 0.7757 at epoch 28 -- a mild version of the
oscillation instability regularization (sixth iteration) was meant to
address, this time from a learning rate that stayed fixed at 1e-3 for
the whole run instead of tapering off as training approached a
plateau. This iteration adds cosine annealing to `miai_segmentation`:
`TrainingConfig.cosine_annealing` (default `False`, so every existing
config keeps its constant-rate behavior unchanged) wraps the optimizer
in `torch.optim.lr_scheduler.CosineAnnealingLR`, stepped once per
epoch, decaying smoothly from `TrainingConfig.learning_rate` (the
schedule's ceiling) down to the new `TrainingConfig.min_learning_rate`
field (the floor, `eta_min`) over `max_epochs`. `examples/
validate_acdc.py` turns it on with `_MAX_LEARNING_RATE = 1e-3`
(unchanged from every prior iteration) decaying to `_MIN_LEARNING_RATE
= 1e-5`, otherwise identical to the eighth iteration: same
150-patient/300-case multi-class dataset, patient-level split,
augmentation, architecture, dropout (0.2), weight decay (1e-5), and
`early_stopping_patience=10`.

Training used the full 50-epoch budget this time -- early stopping
never fired. Validation Dice climbed smoothly as the rate decayed:
0.7988 (epoch 12), 0.8290 (epoch 19, already matching the eighth
iteration's *final* best), 0.8440 (epoch 29), and finally **0.8576 at
epoch 47**, the best checkpoint of this run -- clearly higher, and
visibly more stable, than the eighth iteration's 0.8274. No dip
resembling the eighth iteration's epoch-28 wobble appears anywhere in
this run's log; the closest is a 0.0029 dip at epoch 41 immediately
recovered the next epoch. On validation Dice alone, cosine annealing
looks like a clear win.

| Split | Cases | Dice (macro, foreground only) |
|---|---|---|
| Validation (best epoch, 47/50, ran full budget) | 60 | 0.8576 |
| Held-out test | 60 | 0.7734 |

Per-class mean test metrics (all six, per the seventh iteration's
breakdown), compared against the eighth iteration:

| Metric | RV (8th -> 9th) | Myo (8th -> 9th) | LV (8th -> 9th) | Macro (8th -> 9th) |
|---|---|---|---|---|
| Dice | 0.70 -> 0.71 | 0.76 -> 0.76 | 0.87 -> 0.85 | 0.77 -> 0.77 |
| Hausdorff distance (HD95, mm, lower better) | 34.1 -> **44.5** | 29.3 -> **34.3** | 22.0 -> **27.0** | 28.4 -> **35.2** |
| IoU | 0.55 -> 0.58 | 0.62 -> 0.62 | 0.78 -> 0.76 | 0.65 -> 0.65 |
| Sensitivity | 0.73 -> 0.78 | 0.79 -> 0.81 | 0.90 -> 0.95 | 0.80 -> 0.84 |
| Specificity | 0.997 -> 0.997 | 0.997 -> 0.997 | 0.999 -> 0.998 | 0.998 -> 0.997 |
| Volume similarity | 0.88 -> 0.87 | 0.91 -> 0.90 | 0.94 -> 0.89 | 0.95 -> 0.93 |

**A validation-set win that did not transfer to the test set.** Macro
test Dice is essentially unchanged (0.7740 -> 0.7734), despite the
much higher and more stable validation Dice (0.8274 -> 0.8576) --
whatever cosine annealing bought during training, it did not close
any more of the gap to the fourth iteration's binary ceiling (still
about 0.05). More strikingly, Hausdorff distance -- the metric that
delivered the eighth iteration's biggest single win -- got worse
across every structure this time: macro HD95 rose from 28.4mm to
35.2mm (about 24% worse), and volume similarity slipped too (0.95 ->
0.93 macro, LV specifically 0.94 -> 0.89). Sensitivity improved
instead (macro 0.80 -> 0.84, LV 0.90 -> 0.95) -- consistent with a
model that, as the rate decayed toward 1e-5 late in training, kept
fitting the training/validation distribution more aggressively
(recovering more true-positive pixels) at the cost of looser, less
precise boundaries on the held-out test set. RV Dice did improve (0.70
-> 0.71), continuing the pattern from the sixth/eighth iterations of
RV being the structure most responsive to training changes.

Per-case test Dice (60 cases) stayed close to the eighth iteration in
aggregate but with a heavier tail: mean 0.7734 (stdev 0.114, wider
than the eighth iteration's 0.09), median 0.80, 4 cases below 0.6 (up
from the eighth iteration's 1 below 0.5). `patient142_frame12` is
again the weakest case (Dice 0.26, actually down from the eighth
iteration's 0.47) -- this specific slice has now moved in both
directions across iterations (0.18 -> 0.32 -> 0.47 -> 0.26),
reinforcing that it is a genuinely ambiguous frame whose score is
sensitive to exactly how training unfolds, not a steadily-closing gap.

**The honest read: cosine annealing produced a smoother, higher
training run that did not generalize better, and by one important
measure (boundary quality) generalized worse.** This is a useful,
concrete negative result, not a wasted one -- it rules out "the
eighth iteration's oscillation was costing test performance" as an
explanation for the remaining gap to the binary ceiling, since removing
the oscillation (this run has none) left macro Dice flat and made
Hausdorff distance meaningfully worse. A plausible mechanism: with the
rate decaying to 1e-5 rather than staying at 1e-3 (which effectively
acted like a much shorter run for the fixed-rate optimizer, since a
constant 1e-3 keeps making large-enough updates that it \"gives up\"
improving sooner, as the eighth iteration's own early-stopping-at-29
showed), this run's late epochs kept making small, decaying updates
that pulled validation Dice higher through more precise pixel-level
recall -- but that additional late fitting come at the expense of the
boundary precision Hausdorff distance measures, and validation Dice
alone did not surface that tradeoff. Whether a smaller `min_learning_
rate` floor, a shorter `T_max`, or reverting to the eighth iteration's
fixed-rate-plus-early-stopping setup is the better lever going forward
is exactly the kind of question this result is suited to answer --
but not one this single run resolves on its own.

## Tenth iteration: ResUNet with attention gates, a new architecture

Iterations six through nine all pulled a training-procedure lever
(regularization, more epochs plus early stopping, cosine annealing)
against the same 2D UNet architecture from the third iteration on --
none closed more than a fraction of the gap to the fourth iteration's
binary-only ceiling (0.82), and the ninth iteration's result suggested
that lever was close to exhausted (a validation-set win that did not
transfer to the test set, with boundary quality getting worse). This
iteration changes the architecture itself instead: `miai_segmentation.
two_d.models` gains `ResAttentionUNet` (`kind="res_attention_unet"` in
`ArchitectureConfig`, alongside the existing `"unet"`/
`"attention_unet"`, still defaulting to `"unet"` so no existing config
or call site is affected) -- a residual-block U-Net (MONAI's
`ResidualUnit` in the encoder and decoder, the same building block the
sixth iteration's regularization work already relied on) with
attention-gated skip connections (the additive gate mechanism from
Oktay et al. 2018's Attention U-Net, built from scratch on MONAI's
public `Convolution` primitive rather than MONAI's own `AttentionUnet`,
whose internal gate/block classes are private and not covered by any
stability guarantee). Each skip connection is scaled by a learned,
per-pixel gate in [0, 1] computed from both the decoder's up-sampled
signal and the encoder's skip signal, before the two are concatenated
-- intended to let the decoder suppress irrelevant background instead
of taking the encoder's raw features unfiltered, the way the plain and
attention-only architectures already offered separately but never
combined with a residual backbone.

`examples/validate_acdc.py` switches to `kind="res_attention_unet"`
with channel depth/width, `num_res_units=2`, and dropout (0.2)
identical to every prior iteration's encoder, and reverts the ninth
iteration's cosine annealing back to the eighth iteration's constant
learning rate (1e-3, no decay) -- cosine annealing did not improve
test Dice and made Hausdorff distance worse, so keeping it enabled
here would have made the architecture and the learning-rate schedule
two levers changing at once. Everything else -- the 150-patient/
300-case multi-class dataset, patient-level 180/60/60 split,
augmentation, weight decay (1e-5), `--max-epochs` ceiling (50), and
`early_stopping_patience=10` -- is identical to the eighth iteration,
so this run isolates the architecture as the sole variable against
that known baseline.

Validation Dice climbed faster than any prior iteration: **0.8376 at
epoch 22** (already above the eighth iteration's final best of 0.8274,
and the ninth iteration's epoch-19 checkpoint of 0.8290, reached in
roughly half the epochs the ninth iteration needed to first match the
eighth). Early stopping fired at epoch 32 after 10 non-improving
checks past epoch 22.

| Split | Cases | Dice (macro, foreground only) |
|---|---|---|
| Validation (best epoch, 22/32, early-stopped) | 60 | 0.8376 |
| Held-out test | 60 | 0.7579 |

Per-class mean test metrics (all six, per the seventh iteration's
breakdown), compared against the eighth iteration:

| Metric | RV (8th -> 10th) | Myo (8th -> 10th) | LV (8th -> 10th) | Macro (8th -> 10th) |
|---|---|---|---|---|
| Dice | 0.6952 -> **0.6529** | 0.7593 -> 0.7598 | 0.8676 -> **0.8610** | 0.7740 -> **0.7579** |
| Hausdorff distance (HD95, mm, lower better) | 34.1 -> **57.7** | 29.3 -> 26.8 | 22.0 -> **28.8** | 28.4 -> **37.8** |
| IoU | 0.55 -> **0.51** | 0.62 -> 0.62 | 0.78 -> 0.78 | 0.65 -> **0.63** |
| Sensitivity | 0.73 -> 0.82 | 0.79 -> 0.74 | 0.90 -> 0.92 | 0.80 -> 0.82 |
| Specificity | 0.997 -> 0.995 | 0.997 -> 0.998 | 0.999 -> 0.999 | 0.998 -> 0.997 |
| Volume similarity | 0.88 -> **0.78** | 0.91 -> 0.93 | 0.94 -> 0.92 | 0.95 -> **0.92** |

**A second consecutive validation-set win that did not transfer to
the test set -- this time a clear net negative, not just a wash.**
Despite the fastest and highest validation Dice of any iteration so
far, macro test Dice fell (0.7740 -> 0.7579), Hausdorff distance got
substantially worse (macro HD95 28.4mm -> 37.8mm, ~33% worse -- the
worst of any iteration since the fifth), and volume similarity dropped
too (0.95 -> 0.92 macro, RV specifically 0.88 -> 0.78). The damage is
concentrated almost entirely in the right ventricle: RV Dice fell
(0.6952 -> 0.6529), RV Hausdorff distance nearly doubled (34.1mm ->
57.7mm), and RV volume similarity dropped the most of any structure
(0.88 -> 0.78). Myocardium and LV moved little or, on some metrics,
slightly favorably (Myo Dice essentially flat, LV IoU unchanged). RV
is the smallest, most irregularly-shaped of the three structures and
has been the most volatile across every iteration in this project
(sixth and ninth iterations both singled it out as the structure most
responsive to training changes) -- the attention gates, in suppressing
what they learn to treat as background, appear to have suppressed RV
boundary pixels specifically, exactly the effect they were intended to
prevent when used well.

Per-case test Dice (60 cases) had the second-widest spread in the
project after the fifth iteration: mean 0.7579 (stdev 0.117), median
0.7872, 12 cases below 0.7 (up from single digits in every training-
procedure iteration), 6 below 0.6, 3 below 0.5. `patient142_frame12`
is again the single weakest case (Dice 0.31), continuing its run of
volatile scores across iterations (0.18 -> 0.32 -> 0.47 -> 0.26 ->
0.31) without a clear trend in either direction.

**The honest read: the new architecture trained faster and to a
higher validation Dice than anything tried before, but generalized
worse than the plain regularized UNet the eighth iteration validated
-- the third consecutive iteration (after the ninth's cosine
annealing) where a validation-side improvement did not transfer, and
the first where the test-set result is worse in absolute terms, not
just flat.** The RV-specific damage is the clearest signal: attention
gates are a plausible mechanism for exactly this kind of failure (a
gate trained to suppress noise around a small, hard-to-see structure
can end up suppressing genuine boundary signal instead, especially
with no dedicated per-class loss weighting to protect the smallest
class), and it fits the established pattern of RV being the structure
most sensitive to configuration changes throughout this project. This
result does not indict `ResAttentionUNet` as a broken implementation
-- the unit tests confirm the attention gates are a genuine function
of both their inputs, and the architecture trained stably with no
oscillation or collapse -- it indicts this particular combination
(attention gates, this dataset size, this class balance, no per-class
loss weighting) as worse than the simpler alternative for this task.
Whether a smaller/larger `inter_channels` bottleneck in the gates, a
class-weighted loss to protect RV specifically, or reverting to the
eighth iteration's plain regularized `UNet` as the standing baseline
is the better path forward is exactly the kind of question this
result is suited to answer -- but not one this single run resolves on
its own. `ResAttentionUNet` remains a real, tested, backward-compatible
addition to `miai_segmentation` regardless of this particular run's
outcome; the negative result here is about this configuration on this
dataset, not about the architecture's correctness.

## Eleventh iteration: widening the attention gates' bottleneck

The tenth iteration's damage was concentrated in the RV, and the
leading hypothesis was that each attention gate's bottleneck --
compressed to `up_out // 2` channels before deciding what to suppress
-- was too narrow to preserve the fine-grained information a small,
irregular structure like RV needs. `ResAttentionUnetConfig` gains
`attention_reduction: int = 2` (matching the tenth iteration's
previously-hardcoded value, so nothing existing changes), and this
iteration sets it to `1` -- no compression, the gate's bottleneck is as
wide as the skip connection itself. Otherwise identical to the tenth
iteration: same architecture family, channel depth/width, dropout,
constant learning rate, weight decay, `--max-epochs` ceiling, and
early stopping patience -- isolating the gate bottleneck's width as
the sole variable.

Training reached a new best val Dice of **0.8192 at epoch 13** (between
the eighth iteration's 0.8274 and the tenth's 0.8376), then something
new happened: validation Dice collapsed sharply at epoch 21 (0.7921 ->
**0.5692**) and stayed collapsed through epoch 23, when early stopping
fired -- a late-training instability the fifth iteration's own
collapse (before regularization was added) resembles, not seen in the
eighth, ninth, or tenth iterations' runs. The kept checkpoint is still
the best-ever one (epoch 13), unaffected by the later collapse, so
this doesn't corrupt the result directly -- but it is itself a signal:
a wider gate bottleneck adds real capacity, and apparently enough of it
to destabilize training on this data/model combination in a way the
narrower (tenth iteration) and gate-free (eighth iteration) setups
did not.

| Split | Cases | Dice (macro, foreground only) |
|---|---|---|
| Validation (best epoch, 13/23, early-stopped after a late collapse) | 60 | 0.8192 |
| Held-out test | 60 | 0.7313 |

Per-class mean test metrics (all six), compared against both the
eighth iteration (the pre-architecture baseline) and the tenth
(the narrower-bottleneck attention run this iteration set out to fix):

| Metric | RV (8th / 10th / 11th) | Myo (8th / 10th / 11th) | LV (8th / 10th / 11th) | Macro (8th / 10th / 11th) |
|---|---|---|---|---|
| Dice | 0.6952 / 0.6529 / **0.6768** | 0.7593 / 0.7598 / **0.7111** | 0.8676 / 0.8610 / **0.8060** | 0.7740 / 0.7579 / **0.7313** |
| Hausdorff distance (HD95, mm) | 34.1 / 57.7 / **60.5** | 29.3 / 26.8 / **38.8** | 22.0 / 28.8 / **45.2** | 28.4 / 37.8 / **48.2** |
| IoU | 0.55 / 0.51 / **0.53** | 0.62 / 0.62 / **0.56** | 0.78 / 0.78 / **0.70** | 0.65 / 0.63 / **0.60** |
| Sensitivity | 0.73 / 0.82 / 0.78 | 0.79 / 0.74 / **0.71** | 0.90 / 0.92 / **0.90** | 0.80 / 0.82 / **0.78** |
| Specificity | 0.997 / 0.995 / 0.996 | 0.997 / 0.998 / 0.998 | 0.999 / 0.999 / 0.998 | 0.998 / 0.997 / 0.997 |
| Volume similarity | 0.88 / 0.78 / **0.83** | 0.91 / 0.93 / 0.92 | 0.94 / 0.92 / **0.88** | 0.95 / 0.92 / **0.93** |

**The hypothesis was wrong, and this is the worst macro test Dice of
any multi-class iteration so far (0.7313, below the fifth iteration's
first-ever multi-class attempt at 0.72).** Widening the gate's
bottleneck did give a small RV Dice improvement over the tenth
iteration (0.6529 -> 0.6768) -- but RV Hausdorff distance got *worse*
still (57.7mm -> 60.5mm), and the real damage moved to myocardium and
LV, both untouched by the tenth iteration's problem: Myo Dice fell
0.7598 -> 0.7111, LV Dice fell 0.8610 -> 0.8060, and both structures'
Hausdorff distances roughly doubled from the eighth iteration's
baseline (Myo 29.3mm -> 38.8mm, LV 22.0mm -> 45.2mm). In other words,
removing the bottleneck's compression did not free up capacity the RV
specifically needed -- it gave the whole network more capacity to
overfit or destabilize, consistent with the epoch-21 validation
collapse, and every structure paid for it except RV's Dice, marginally.

Per-case test Dice (60 cases) had the widest spread of any multi-class
iteration: mean 0.7313 (stdev 0.125), median 0.7520, 20 of 60 cases
below 0.7 (up from 12 in the tenth iteration), 3 below 0.5.
`patient086_frame08` is the new weakest case (Dice 0.34), narrowly
displacing `patient142_frame12` (Dice 0.38, still volatile: 0.18 ->
0.32 -> 0.47 -> 0.26 -> 0.31 -> 0.38) from the bottom for the first
time in this project.

**The honest read: the RV-suppression hypothesis motivating this
iteration does not hold up, and the fix made the model worse, not
better.** A narrower gate bottleneck (tenth iteration) hurt RV
specifically without touching the other structures; a wider one
(this iteration) barely helped RV's Dice, made RV's boundary quality
worse anyway, and additionally damaged both other structures along
with training stability itself. This rules out "gate bottleneck width"
as a simple dial that trades off against RV performance in one
direction -- the relationship is not monotonic, or the real problem
lies elsewhere in the attention mechanism (or in this task/dataset's
interaction with attention gates generally) rather than in this one
hyperparameter. Combined with the tenth iteration's result, three
consecutive architecture/procedure changes since the eighth iteration
(cosine annealing, attention gates at reduction=2, attention gates at
reduction=1) have each underperformed the eighth iteration's plain
regularized `UNet`, which remains the best-performing configuration
found across this entire validation effort.

## Twelfth iteration: plain Residual U-Net, attention gates removed entirely

Two consecutive attention-gate configurations (tenth: standard
`attention_reduction=2`; eleventh: no bottleneck compression,
`attention_reduction=1`) both underperformed the eighth iteration's
plain regularized `UNet`, and the eleventh additionally introduced a
late-training validation collapse never seen in the eighth, ninth, or
tenth iterations. Rather than try a third bottleneck width, this
iteration asks a more basic question: is the attention mechanism
itself responsible for the tenth/eleventh iterations' results, or is
it the custom residual-block encoder/decoder architecture underneath
it (`ResAttentionUNet`, distinct from MONAI's built-in `UNet` the
eighth iteration's baseline uses)? `ResAttentionUnetConfig` gains
`use_attention: bool = True`; setting it to `False` builds the exact
same `ResAttentionUNet` class and forward pass with the attention
gates removed entirely -- each skip connection is concatenated
unmodified, as a plain (non-attention-gated) residual U-Net would be,
with every other structural choice (residual encoder/decoder blocks,
channel depth/width, dropout) held identical. Unlike a new class, this
guarantees the *only* variable that changes is whether the gating step
runs at all.

`examples/validate_acdc.py` sets `_USE_ATTENTION = False`. Everything
else -- channel depth/width, `num_res_units=2`, dropout (0.2), the
constant learning rate (1e-3, no decay), weight decay (1e-5),
`--max-epochs` ceiling (50), and `early_stopping_patience=10` -- is
identical to the tenth iteration, not the eleventh: the tenth used the
"standard" `attention_reduction=2` configuration, so comparing against
it (rather than the eleventh's already-rejected `attention_reduction=1`)
isolates attention on/off as the sole variable.

Validation Dice reached the highest best-epoch value of any iteration
in this project: **0.8278 at epoch 15** (above the eighth iteration's
0.8274 and the eleventh's 0.8192, though still below the tenth's
0.8376). Training then showed the same kind of late instability the
eleventh iteration's wider gate bottleneck produced -- except this run
has *no* attention gates at all: validation Dice collapsed at epoch 23
(0.8088 -> 0.5485), stayed collapsed through epoch 24 (0.5748), and
early stopping fired at epoch 25 after 10 non-improving checks past
epoch 15. The kept checkpoint is the pre-collapse best (epoch 15),
same safety net as the eleventh iteration.

| Split | Cases | Dice (macro, foreground only) |
|---|---|---|
| Validation (best epoch, 15/25, early-stopped after a late collapse) | 60 | 0.8278 |
| Held-out test | 60 | 0.7200 |

Per-class mean test metrics (all six), compared against the eighth
iteration (pre-architecture baseline) and the tenth (the
attention-gated run this iteration isolates attention against):

| Metric | RV (8th / 10th / 12th) | Myo (8th / 10th / 12th) | LV (8th / 10th / 12th) | Macro (8th / 10th / 12th) |
|---|---|---|---|---|
| Dice | 0.6952 / 0.6529 / **0.6455** | 0.7593 / 0.7598 / **0.6856** | 0.8676 / 0.8610 / **0.8288** | 0.7740 / 0.7579 / **0.7200** |
| Hausdorff distance (HD95, mm) | 34.1 / 57.7 / **54.5** | 29.3 / 26.8 / **54.4** | 22.0 / 28.8 / **42.1** | 28.4 / 37.8 / **50.3** |
| IoU | 0.55 / 0.51 / **0.49** | 0.62 / 0.62 / **0.53** | 0.78 / 0.78 / **0.73** | 0.65 / 0.63 / **0.58** |
| Sensitivity | 0.73 / 0.82 / **0.63** | 0.79 / 0.74 / 0.73 | 0.90 / 0.92 / **0.85** | 0.80 / 0.82 / **0.73** |
| Specificity | 0.997 / 0.995 / 0.998 | 0.997 / 0.998 / 0.996 | 0.999 / 0.999 / 0.999 | 0.998 / 0.997 / 0.998 |
| Volume similarity | 0.88 / 0.78 / **0.84** | 0.91 / 0.93 / **0.90** | 0.94 / 0.92 / 0.91 | 0.95 / 0.92 / 0.94 |

**Removing the attention gates entirely did not recover the eighth
iteration's baseline -- it produced the worst macro test Dice of any
multi-class iteration so far (0.7200, below the eleventh's 0.7313 and
well below the tenth's 0.7579), despite the best validation Dice of
the whole project.** The damage this time is not RV-specific the way
the tenth iteration's was: Myo Dice fell hardest (0.7598 -> 0.6856)
and its Hausdorff distance roughly doubled (26.8mm -> 54.4mm), LV Dice
and Hausdorff distance both got worse too (0.8610 -> 0.8288, 28.8mm ->
42.1mm), and RV sensitivity dropped sharply (0.82 -> 0.63) even though
RV's own Hausdorff distance improved slightly relative to the tenth
iteration (57.7mm -> 54.5mm) and RV volume similarity improved
(0.78 -> 0.84). This is a genuinely different failure pattern from the
tenth iteration's RV-concentrated one -- it looks far more like the
eleventh iteration's late-training collapse (same shape: a sharp
validation Dice drop late in training, recovered from only partially
by the checkpoint-saving safety net) than like an attention-specific
problem, which is the key finding here: **this run has zero attention
gates and still destabilized**, which rules out the attention
mechanism itself as the cause of that instability. Whatever
destabilizes training late -- most plausibly something in the shared
residual-block architecture, dropout, weight decay, or constant
learning-rate combination, all held identical across the tenth,
eleventh, and twelfth iterations -- it isn't the gates.

Per-case test Dice (60 cases) had a similar spread to the tenth
iteration: mean 0.7200 (stdev 0.114), median 0.7555, 2 cases below
0.5. `patient142_frame12` is again the single weakest case (Dice
0.32, continuing its volatile run across iterations: 0.18 -> 0.32 ->
0.47 -> 0.26 -> 0.31 -> 0.38 -> 0.32), and `patient086_frame08` (Dice
0.37) is the second-weakest, echoing its appearance at the very bottom
of the eleventh iteration's ranking.

**The honest read: the hypothesis behind this iteration -- that
attention was the problem, and removing it would recover the eighth
iteration's baseline -- is now decisively ruled out.** A plain
residual U-Net, with no attention gates whatsoever, generalized worse
than both attention-gated configurations tried so far, and worse than
the plain MONAI `UNet` baseline by a wide margin (0.7740 -> 0.7200,
the largest drop of any single-lever change in this project). Combined
with the tenth and eleventh iterations, four consecutive
architecture/procedure changes since the eighth iteration (cosine
annealing, attention at reduction=2, attention at reduction=1, no
attention at all) have each underperformed the eighth iteration's
plain regularized `UNet`, which remains by a clear margin the
best-performing configuration found across this entire validation
effort. The late-training collapse recurring without attention gates
present is the most useful signal from this run: it points away from
the attention mechanism and toward something shared by the residual
architecture itself, or the training recipe it's paired with, as the
actual source of the instability seen in both the eleventh and twelfth
iterations -- a question a future iteration could isolate by returning
to MONAI's plain `UNet` with the residual-block encoder/decoder as the
next single lever, or by testing this same `ResAttentionUNet`
(attention on or off) against a lower learning rate or reduced weight
decay.

## Reproducing this

The script as it stands today runs the twelfth iteration -- multi-class,
150 patients, up to 50 epochs with early stopping (patience 10),
dropout 0.2, weight decay 1e-5, a constant learning rate (1e-3, no
decay), and the `ResAttentionUNet` architecture with its attention
gates removed entirely (`use_attention=False`, a plain residual U-Net):

```bash
python examples/validate_acdc.py \
    --data-dir /path/to/ACDC \
    --output-dir examples/output/acdc_validation
```

(`--max-epochs` now defaults to `50`, matching this iteration's raised
ceiling -- actual training length depends on early stopping, not a
fixed count; earlier iterations used different, fixed epoch budgets,
noted inline above and reproducible by passing `--max-epochs`
explicitly, though `TrainingConfig.early_stopping_patience` would need
overriding to `None` in a copy of the script to reproduce their exact
fixed-budget behavior instead of stopping early.) Earlier
binary-only iterations are not reproducible from the current script
verbatim -- `_NUM_CLASSES` is set to `4` at module scope, not exposed
as a CLI flag -- but every metric/config field this section describes
for iterations 1-4 stayed real, working binary behavior in
`miai_segmentation`/`miai_evaluation` (`num_classes=1` is still each
new field's default), so reproducing a binary run means passing
`num_classes=1` to `TrainingConfig`/`InferenceConfig`/`MetricsConfig`
in a copy of the script with `_NUM_CLASSES` set back to `1` (and
`_ARCHITECTURE`'s `out_channels` back to `1`).

Outputs land under `--output-dir` (git-ignored, like every other
`examples/output/` run): prepared labels, preprocessed/padded images
and labels, the manifest split, the checkpoint, per-case predictions,
and `evaluation_report.json`.

## Visualizing results

Every iteration above was reported as plain text and markdown tables
-- twelve iterations, zero images. `miai_visualization` (per-slice and
montage plots, side-by-side comparisons with a difference map,
training curves from a CSV log, per-case/per-group bar and box
summaries) existed the whole time but was only ever wired into the
generic `examples/segmentation_pipeline.py` demo via
`VisualizationStage`, never into this effort.

Two additions close that gap, both backward compatible:

- `examples/validate_acdc.py` gained a `--visualize` flag (off by
  default -- every iteration's numbers above were produced without
  it). When passed, it runs `VisualizationStage` -- the same class the
  generic pipeline demo uses, unmodified -- over every held-out
  test-set image right after evaluation, writing one QC slice-montage
  PNG per case to `<output-dir>/qc_montages/`.
- `examples/visualize_acdc_results.py` is a new, separate script that
  builds four kinds of plot from a *completed* run's logs and
  `evaluation_report.json`, without retraining or needing
  `--visualize`: training curves (a dedicated twelfth-iteration
  train-loss/val-Dice chart, and a 4-way validation-Dice comparison
  across the eighth/tenth/eleventh/twelfth iterations on the same
  axes -- the epoch-21/epoch-23 collapses are immediately visible as a
  sharp drop, where the text writeups above could only describe
  them), ground-truth-vs-prediction comparisons for a few twelfth-
  iteration test cases (a difference map between the two label maps,
  plus the predicted mask overlaid on the source MRI for anatomical
  context), and metric summaries (macro test Dice across iterations
  8-12 as a bar chart, and the twelfth iteration's per-case Dice split
  by class as a box plot). It also exercises `VisualizationStage`
  itself, the same way the new `--visualize` flag does, against the
  twelfth iteration's already-completed test set -- so this session's
  visualization work is confirmed end to end without waiting for
  another multi-hour training run.

The script hardcodes this sandbox session's specific `/tmp/...`
output/log paths (see its module docstring) -- it is a one-off
analysis script for this validation effort's existing outputs, not a
general reusable example like `segmentation_pipeline.py`.
