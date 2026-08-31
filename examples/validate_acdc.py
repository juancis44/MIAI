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

**Sixth iteration: explicit regularization (dropout, weight decay).**
The fifth iteration's training run hit a late, transient instability
-- validation Dice collapsed to 0.0 at epoch 23 (loss spiking
0.14 -> 0.18 -> 0.44), a training-loop failure mode explicit
regularization is meant to guard against, and a concrete, real-data
signal (not just a theoretical gap) that this lever was worth pulling
next. Two new, orthogonal knobs, both newly added to
:mod:`miai_segmentation` and both defaulting to off (``0.0``) so every
prior iteration's config keeps working unchanged: ``UNetConfig.
dropout`` (activation dropout inside each residual unit's ADN block,
set here to ``_DROPOUT = 0.2``) and ``TrainingConfig.weight_decay``
(L2 penalty on the weights themselves, passed straight to
``torch.optim.Adam``, set here to ``_WEIGHT_DECAY = 1e-5``). Otherwise
identical to the fifth iteration: same full 150-patient/300-case
multi-class dataset, patient-level split, augmentation, architecture
depth, and 25-epoch budget -- so any change in the result isolates the
effect of regularization, not a confound from also changing the data
or architecture. See ``docs/real_data_validation.md`` for the result.

**Seventh iteration: per-class breakdown for every metric, not just
Dice.** The fifth iteration added a per-class ``dice_class_{c}``
breakdown, but every other opted-in metric (Hausdorff distance, IoU,
sensitivity, specificity, volume similarity) still only reported one
macro-averaged number even in multi-class mode -- hiding, e.g.,
whether the RV's lower Dice comes with a correspondingly worse
Hausdorff distance (a genuinely worse boundary) or is driven mostly by
size/overlap rather than boundary shape. :func:`miai_evaluation.
metrics.compute_case_metrics` now reports a ``{metric}_class_{c}``
entry for every opted-in metric in multi-class mode, the same pattern
``dice_class_{c}`` already used -- ``hausdorff_distance_class_{c}``,
``iou_class_{c}``, ``sensitivity_class_{c}``, ``specificity_
class_{c}``, and ``volume_similarity_class_{c}``, each computed on
that class's one-hot channel alone, not derived from the macro
average. No new data, no new training run -- this iteration only
changes what ``compute_case_metrics`` reports for the same predictions
the sixth iteration already produced, so ``run_validation`` re-scores
the sixth iteration's checkpoint against the (unchanged) evaluation
config rather than re-training. ``named_class_metrics`` (formerly
``named_class_dice``) now covers all six metrics via
``_PER_CLASS_METRIC_PREFIXES``, not just Dice. See
``docs/real_data_validation.md`` for the result.

**Eighth iteration: more epochs with early stopping, chasing the
binary-era test Dice ceiling.** The sixth iteration's best checkpoint
landed early (epoch 13 of a fixed 25-epoch budget) and val Dice never
improved again in the remaining 12 epochs -- a fixed epoch budget has
no way to tell whether that's a genuine plateau or just bad luck
within too short a run. This iteration adds early stopping to
:mod:`miai_segmentation`, a real (if small) feature addition:
``TrainingConfig.early_stopping_patience`` (default ``None``, so every
prior iteration's config keeps behaving exactly as before), which
stops training once validation Dice has gone this many consecutive
validation checks without a new best. ``--max-epochs`` is raised to 50
(from 25) so a later improvement gets a real chance to surface, and
``_EARLY_STOPPING_PATIENCE = 10`` bounds how long training keeps
running once it's actually plateaued -- neither number alone would do
what the two together do: a fixed higher budget risks wasting hours
training past the point of any real improvement, and a short patience
without a raised budget wouldn't have let this question get asked at
all. Otherwise identical to the sixth iteration: same full
150-patient/300-case multi-class dataset, patient-level split,
augmentation, architecture depth, dropout, and weight decay -- so any
change in the result isolates the effect of training longer with early
stopping, not a confound from also changing the data, architecture, or
other regularization. The goal: close some of the gap between this
multi-class series' best result (0.75, sixth iteration) and the
fourth iteration's binary-only ceiling (0.82) -- multi-class is
expected to stay below that ceiling for the reasons the fifth
iteration already laid out (getting the class right, not just the
pixel, is a strictly harder task), but it's an open question how much
of the remaining 0.07 gap is architecture/data-scale-limited versus
simply under-trained. See ``docs/real_data_validation.md`` for the
result.

**Ninth iteration: cosine-annealed learning rate.** Every iteration
through the eighth used a single, constant Adam learning rate for the
whole run -- the eighth iteration's own training curve hints this may
have left something on the table: its later epochs oscillated around
its plateau (e.g. epoch 19's best of 0.8274 followed by epoch 28's
dip to 0.7757) rather than settling smoothly, a pattern consistent
with a fixed step size that is well-suited to early progress but too
large once the model is closer to convergence. This iteration adds a
learning rate schedule to :mod:`miai_segmentation`, a real (if small)
feature addition: ``TrainingConfig.cosine_annealing`` (default
``False``, so every prior iteration's config keeps behaving exactly
as before) wraps the optimizer in ``torch.optim.lr_scheduler.
CosineAnnealingLR``, stepped once per epoch, smoothly decaying the
rate from ``TrainingConfig.learning_rate`` down to a new floor,
``TrainingConfig.min_learning_rate`` (default ``0.0``), following a
cosine curve over ``max_epochs``. This iteration sets both ends of
that curve explicitly -- ``_MAX_LEARNING_RATE = 1e-3`` (the same
constant rate every prior iteration used, so the schedule starts
exactly where the proven-working setup already was) and
``_MIN_LEARNING_RATE = 1e-5`` (two orders of magnitude lower, small
enough to let the model fine-tune gently in later epochs without
letting the rate collapse to a standstill). Otherwise identical to the
eighth iteration: same full 150-patient/300-case multi-class dataset,
patient-level split, augmentation, architecture depth, dropout, weight
decay, raised ``--max-epochs`` ceiling, and early stopping patience --
so any change in the result isolates the effect of the schedule, not a
confound from also changing the data, architecture, other
regularization, or the epoch budget. The goal, same as the eighth
iteration's: close more of the remaining gap to the fourth iteration's
binary-only ceiling (0.82), this time via a different lever than more
epochs alone, per the eighth iteration's own diagnosis that further
epochs at a constant rate were unlikely to help much more. See
``docs/real_data_validation.md`` for the result -- a validation-set
improvement that did not transfer to the test set (macro Dice
essentially unchanged, Hausdorff distance meaningfully worse).

**Tenth iteration: ResUNet with attention gates, a new architecture.**
Every iteration through the ninth used the same plain-residual UNet
(``ArchitectureConfig(kind="unet")``) -- a different, orthogonal lever
from the training-procedure changes (regularization, early stopping,
LR scheduling) the sixth through ninth iterations each tried. This
iteration adds a third architecture to :mod:`miai_segmentation.two_d`,
``ResAttentionUNet`` (``kind="res_attention_unet"``): the same residual
encoder/decoder blocks the plain UNet uses, plus attention gates on
every skip connection (Oktay et al. 2018's mechanism, the same one
``AttentionUnetConfig`` already offered on top of *plain* convolutions,
now combined with residual blocks instead) -- a real, backward-compatible
addition to :mod:`miai_segmentation` (see ``docs/real_data_validation.md``
and the CHANGELOG for the library-level change). Since the ninth
iteration's cosine annealing turned out not to help (see above), this
iteration reverts ``TrainingConfig.cosine_annealing`` to its default
``False`` -- ``_MAX_LEARNING_RATE`` is the single, constant rate every
iteration except the ninth has used -- so architecture is the *only*
variable that changes versus the eighth iteration: same full
150-patient/300-case multi-class dataset, patient-level split,
augmentation, channel depth/width, dropout, weight decay, raised
``--max-epochs`` ceiling, and early stopping patience. The goal: test
whether attention gates on a residual backbone close more of the
remaining gap to the fourth iteration's binary-only ceiling (0.82) than
the training-procedure levers the sixth through ninth iterations tried.
See ``docs/real_data_validation.md`` for the result -- the fastest and
highest validation Dice of any iteration so far, but the first with a
worse (not just flat) test-set result, concentrated almost entirely in
the right ventricle (RV Dice, Hausdorff distance, and volume
similarity all got meaningfully worse; myocardium and LV barely moved).

**Eleventh iteration: widening the attention gates' bottleneck.** The
tenth iteration's damage was concentrated in the RV -- the smallest,
most irregularly-shaped of the three structures, and (per the sixth,
eighth, and ninth iterations) already the one most sensitive to
configuration changes throughout this project. A plausible mechanism:
each attention gate's ``1x1`` projections (the gating signal and the
skip signal) were compressed down to ``up_out // 2`` channels
(:class:`~miai_segmentation.two_d.models._AttentionGate`'s
``inter_channels``) before deciding what to suppress -- a narrower
bottleneck than the skip connection itself, which may have discarded
exactly the fine-grained information a gate needs to tell "genuine RV
boundary" apart from "background" on a structure this small. This
iteration adds ``ResAttentionUnetConfig.attention_reduction`` to
:mod:`miai_segmentation` (default ``2``, matching the tenth iteration's
previously-hardcoded bottleneck exactly, so that run's config is still
reproducible byte-for-byte) and sets it to ``_ATTENTION_REDUCTION = 1``
here -- removing the compression entirely, so each gate's bottleneck is
as wide as the skip connection it's gating. Otherwise identical to the
tenth iteration: same architecture family (``kind="res_attention_
unet"``), channel depth/width, ``num_res_units``, dropout, constant
learning rate, weight decay, ``--max-epochs`` ceiling, and early
stopping patience -- so any change in the result isolates the effect of
the gate bottleneck's width, not a confound from also changing the
data, the rest of the architecture, or the training procedure. The
goal: recover the RV Dice/Hausdorff distance the tenth iteration lost,
without losing the fast, high validation Dice attention gates already
demonstrated they're capable of. See ``docs/real_data_validation.md``
for the result.

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
from miai_segmentation.two_d.models import ArchitectureConfig, ResAttentionUnetConfig
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
#: compute_case_metrics`'s generic ``{metric}_class_{c}`` keys -- kept
#: here, not in :mod:`miai_evaluation`, since that module stays
#: dataset-agnostic on purpose (see ``MetricsConfig.num_classes``'s
#: docstring) and this mapping is ACDC-specific domain knowledge.
_CLASS_NAMES = {1: "RV", 2: "Myo", 3: "LV"}

#: Metric name prefixes :func:`miai_evaluation.metrics.
#: compute_case_metrics` reports a ``{prefix}_class_{c}`` per-class
#: breakdown for, in multi-class mode -- see the module docstring's
#: "Seventh iteration" section. Kept as one list here so
#: ``run_validation``'s ``named_class_metrics`` mapping covers every
#: metric this run opts into automatically, without hardcoding "dice"
#: as the only one that gets a per-class name.
_PER_CLASS_METRIC_PREFIXES = (
    "dice",
    "hausdorff_distance",
    "iou",
    "sensitivity",
    "specificity",
    "volume_similarity",
)

#: Dropout probability inside each residual unit's ADN block -- see the
#: module docstring's "Sixth iteration" section. ``0.2`` is a
#: conventional light-to-moderate choice for a UNet this size (deep
#: enough to have real capacity to overfit at this data scale, shallow
#: enough that heavier dropout would likely just slow convergence
#: rather than help).
_DROPOUT = 0.2

#: Adam L2 weight decay -- see the module docstring's "Sixth iteration"
#: section. ``1e-5`` is a conventional light default: small enough not
#: to fight the primary Dice loss signal, large enough to discourage
#: the kind of large-weight excursion the fifth iteration's epoch-23
#: instability looked like.
_WEIGHT_DECAY = 1e-5

#: Consecutive validation checks with no val Dice improvement before
#: training stops early -- see the module docstring's "Eighth
#: iteration" section. The sixth iteration's best checkpoint landed at
#: epoch 13 of 25 and never improved again in the 12 epochs that
#: followed, which is what a fixed epoch budget can't see coming: this
#: iteration raises ``--max-epochs`` well past 25 so a later
#: improvement gets a real chance to show up, while ``10`` bounds how
#: long training keeps running once it's actually plateaued, instead of
#: burning the full raised budget on a run that stopped improving long
#: before it.
_EARLY_STOPPING_PATIENCE = 10

#: The constant Adam learning rate every iteration except the ninth
#: has used -- see the module docstring's "Ninth iteration" section for
#: why the ninth iteration's cosine-annealed alternative
#: (``TrainingConfig.cosine_annealing``/``.min_learning_rate``, both
#: still real, backward-compatible features of :mod:`miai_segmentation`)
#: is not enabled here: it did not improve test Dice and made Hausdorff
#: distance meaningfully worse, so the tenth iteration reverts to this
#: constant rate to isolate the architecture change below as the only
#: variable versus the eighth iteration.
_MAX_LEARNING_RATE = 1e-3

#: Attention gate bottleneck divisor -- see the module docstring's
#: "Eleventh iteration" section. ``1`` disables the compression
#: entirely (``ResAttentionUnetConfig.attention_reduction``'s default,
#: ``2``, is what the tenth iteration actually ran with, hardcoded at
#: the time); this iteration's sole variable versus the tenth.
_ATTENTION_REDUCTION = 1

#: 2D per-slice architecture (see the module docstring's "Third
#: iteration" section for why 2D, not 3D, is the right fit for this
#: data): ``kind="res_attention_unet"``, new in the tenth iteration
#: (see that section) -- same channel depth/width every prior iteration
#: used (16->32->64->128, three stride-2 levels, 2 residual units per
#: level), now with attention-gated skip connections on top of the
#: residual blocks. ``out_channels=_NUM_CLASSES`` (up from the binary
#: iterations' implicit ``1``) is what actually makes this a
#: multi-class model -- see the module docstring's "Fifth iteration"
#: section for how ``TrainingConfig``/``InferenceConfig``/
#: ``MetricsConfig`` pick up the same ``_NUM_CLASSES`` to train, infer,
#: and score consistently as 4-class instead of binary. ``dropout=
#: _DROPOUT`` is unchanged from the sixth iteration.
#: ``attention_reduction=_ATTENTION_REDUCTION`` is new in the eleventh
#: iteration -- see the module docstring's "Eleventh iteration" section.
_ARCHITECTURE = SegmentationModalityConfig(
    modality="two_d",
    two_d=ArchitectureConfig(
        kind="res_attention_unet",
        res_attention_unet=ResAttentionUnetConfig(
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            out_channels=_NUM_CLASSES,
            dropout=_DROPOUT,
            attention_reduction=_ATTENTION_REDUCTION,
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
                learning_rate=_MAX_LEARNING_RATE,
                weight_decay=_WEIGHT_DECAY,
                early_stopping_patience=_EARLY_STOPPING_PATIENCE,
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
    # {metric}_class_{c} keys -- see _CLASS_NAMES for why this mapping
    # lives here, not in miai_evaluation. Covers every metric this run
    # opted into (dice, hausdorff_distance, iou, sensitivity,
    # specificity, volume_similarity), not just Dice -- see the module
    # docstring's "Seventh iteration" section.
    named_class_metrics = {
        f"{metric_prefix}_{name.lower()}": metrics["mean"][f"{metric_prefix}_class_{class_id}"]
        for metric_prefix in _PER_CLASS_METRIC_PREFIXES
        for class_id, name in _CLASS_NAMES.items()
        if f"{metric_prefix}_class_{class_id}" in metrics["mean"]
    }
    if named_class_metrics:
        logger.info("Per-class mean test metrics: %s", named_class_metrics)

    return {
        "manifest_sizes": {k: len(v) for k, v in manifest.items()},
        "checkpoint": context.require("model_checkpoint_path"),
        "mean_metrics": metrics["mean"],
        "named_class_metrics": named_class_metrics,
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
        default=50,
        help="50 is the eighth iteration's raised ceiling, meant to be cut short by "
        "early stopping (TrainingConfig.early_stopping_patience, see the module "
        "docstring's 'Eighth iteration' section) rather than always fully used. 25 is "
        "what the fourth/fifth/sixth iterations used; the third iteration's smaller "
        "50-patient subset used 40 -- pass explicitly to match any of those.",
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
