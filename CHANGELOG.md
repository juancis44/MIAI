# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Explicit regularization (dropout, weight decay)**: `miai_segmentation.
  {two_d,three_d}.models.UNetConfig` gains a `dropout` field (default
  `0.0`, matching MONAI `UNet`'s own default -- wired straight through
  to its `dropout` constructor arg), and
  `miai_segmentation.three_d.train.TrainingConfig` (shared by `two_d`)
  gains a `weight_decay` field (default `0.0`, matching
  `torch.optim.Adam`'s own default -- wired straight through to its
  `weight_decay` kwarg). Both default to off, so every existing config
  or call site keeps working byte-for-byte unchanged.
- Sixth ACDC validation iteration (2026-08-28): `examples/validate_acdc.py`
  puts the new regularization support to use (`dropout=0.2`,
  `weight_decay=1e-5`), otherwise identical to the fifth iteration
  (full 150-patient/300-case multi-class dataset, 2D per-slice UNet,
  patient-level 180/60/60 split, augmentation, 25 epochs). Motivated by
  the fifth iteration's late training instability (validation Dice
  collapsed to 0.0 at epoch 23). Result: the instability is gone (no
  collapse anywhere in this run), and macro test Dice improved from
  0.72 to **0.75**, driven mostly by the right ventricle (RV Dice 0.58
  -> 0.70) -- the structure the fifth iteration had flagged as hardest.
  Full writeup in `docs/real_data_validation.md`.
- **Per-class breakdown for every metric, not just Dice**:
  `miai_evaluation.metrics.compute_case_metrics` now reports a
  `{metric}_class_{c}` entry for every opted-in metric in multi-class
  mode (`hausdorff_distance_class_{c}`, `iou_class_{c}`,
  `sensitivity_class_{c}`, `specificity_class_{c}`, `volume_
  similarity_class_{c}`), not just `dice_class_{c}` as before. Each is
  computed on that class's one-hot channel alone, the same way
  `dice_class_{c}` already was. Backward compatible: the new keys only
  appear when both `num_classes > 1` and the corresponding `include_*`
  flag is set, so no existing single-class or macro-only caller is
  affected. Seventh ACDC validation iteration (2026-08-28) re-scores
  the sixth iteration's checkpoint with the expanded metric config (no
  retraining): the RV-is-hardest story from Dice mostly holds for
  IoU/sensitivity too, but volume similarity tells a different story --
  RV (0.87) is its *worst*-scoring structure, not LV, and Myo (0.91) is
  its best, showing Dice and volume similarity can disagree about which
  structure is hardest. Full writeup in `docs/real_data_validation.md`.

- **Early stopping**: `miai_segmentation.three_d.train.TrainingConfig`
  (shared by `two_d`) gains an `early_stopping_patience: int | None`
  field (default `None`, unchanged behavior -- training always runs
  the full `max_epochs`). When set, training stops once validation
  Dice has gone that many consecutive validation checks without a new
  best, and checkpoints the best epoch seen, same as always.
- Eighth ACDC validation iteration (2026-08-28): `examples/
  validate_acdc.py` raises `--max-epochs` from 25 to 50 and sets
  `early_stopping_patience=10`, motivated by the sixth iteration's best
  checkpoint landing early (epoch 13 of 25) with no further improvement
  in the remaining 12 epochs. Otherwise identical to the sixth
  iteration. Training used the extra room -- val Dice kept improving to
  a new best of 0.8274 at epoch 19, then stopped at epoch 29 after 10
  non-improving checks. Result: macro test Dice rose from 0.75 to
  **0.77**, with the largest single improvement in Hausdorff distance
  (macro HD95 46.2mm -> 28.4mm, ~39% better). Narrowed the gap to the
  fourth iteration's binary-only ceiling (0.82) from 0.10 to 0.05. Full
  writeup in `docs/real_data_validation.md`.

- **Cosine-annealed learning rate**: `miai_segmentation.three_d.
  train.TrainingConfig` (shared by `two_d`) gains a `cosine_annealing:
  bool` field (default `False`, unchanged constant-rate behavior) and
  a `min_learning_rate: float` field (default `0.0`, matching
  `CosineAnnealingLR`'s own `eta_min` default). When enabled, the
  optimizer is wrapped in `torch.optim.lr_scheduler.CosineAnnealingLR`
  (`T_max=max_epochs`, `eta_min=min_learning_rate`), stepped once per
  epoch, decaying smoothly from `learning_rate` (the ceiling) to
  `min_learning_rate` (the floor) instead of holding one constant rate
  for the whole run.
- Ninth ACDC validation iteration (2026-08-30): `examples/
  validate_acdc.py` turns cosine annealing on (`_MAX_LEARNING_RATE =
  1e-3` unchanged, decaying to `_MIN_LEARNING_RATE = 1e-5`), motivated
  by the eighth iteration's late-training oscillation (a dip to 0.7757
  at epoch 28). Otherwise identical to the eighth iteration. Training
  ran the full 50-epoch budget with no early stopping and no
  oscillation, reaching a new best val Dice of 0.8576 (up from 0.8274)
  at epoch 47. Result: macro test Dice is essentially unchanged (0.7740
  -> 0.7734), and Hausdorff distance -- the eighth iteration's biggest
  single win -- got worse across every class (macro HD95 28.4mm ->
  35.2mm, ~24% worse); sensitivity improved instead (macro 0.80 ->
  0.84). A validation-set improvement that did not transfer to the
  test set, and by one measure (boundary quality) generalized worse.
  Full writeup in `docs/real_data_validation.md`.

- **ResUNet with attention gates**: `miai_segmentation.two_d.models`
  gains `ResAttentionUNet` and a new `ArchitectureConfig` kind,
  `"res_attention_unet"` (alongside the existing `"unet"`/
  `"attention_unet"`, still defaulting to `"unet"` -- no existing
  config or call site is affected). Combines MONAI's `ResidualUnit` in
  the encoder/decoder with an additive attention gate (Oktay et al.
  2018) on every skip connection, built from scratch on MONAI's public
  `Convolution` primitive since MONAI's own `AttentionUnet` internals
  are private. Neither MIAI nor MONAI previously offered residual
  blocks and attention gates together in one architecture.
- Tenth ACDC validation iteration (2026-08-30/31): `examples/
  validate_acdc.py` switches to `kind="res_attention_unet"` and
  reverts the ninth iteration's cosine annealing back to a constant
  learning rate (1e-3), isolating the new architecture as the sole
  variable against the eighth iteration's baseline. Training reached a
  new-fastest best val Dice of 0.8376 at epoch 22 (early-stopped at
  epoch 32). Result: macro test Dice fell (0.7740 -> **0.7579**) and
  Hausdorff distance got substantially worse (macro HD95 28.4mm ->
  **37.8mm**, ~33% worse), with the damage concentrated in the right
  ventricle (RV Dice 0.6952 -> 0.6529, RV HD95 34.1mm -> 57.7mm, RV
  volume similarity 0.88 -> 0.78). A validation-set win that not only
  failed to transfer but generalized worse than the plain regularized
  UNet -- the first iteration with a worse, not just flat, test-set
  result. Full writeup in `docs/real_data_validation.md`.

- **Configurable attention-gate bottleneck**: `ResAttentionUnetConfig`
  gains `attention_reduction: int = 2` (matching the tenth iteration's
  previously-hardcoded `up_out // 2` bottleneck width, so no existing
  config or call site is affected) -- controls how much each attention
  gate compresses its gate/skip projections before deciding what to
  suppress; `1` disables the compression entirely.
- Eleventh ACDC validation iteration (2026-08-31): `examples/
  validate_acdc.py` sets `attention_reduction=1`, motivated by the
  tenth iteration's RV-specific damage and the hypothesis that a
  narrower bottleneck had discarded fine-grained information RV
  needed. Otherwise identical to the tenth iteration. Training hit a
  late validation-Dice collapse (0.79 -> 0.57 at epoch 21, not seen in
  the eighth/ninth/tenth iterations), early-stopping at epoch 23 with
  its best checkpoint from epoch 13 (val Dice 0.8192). Result: the
  hypothesis did not hold -- macro test Dice fell further (0.7579 ->
  **0.7313**, the worst multi-class result in this project), RV
  Hausdorff distance got worse, not better (57.7mm -> 60.5mm), and the
  real damage moved to myocardium and LV (Myo Dice 0.7598 -> 0.7111,
  LV Dice 0.8610 -> 0.8060), both untouched by the tenth iteration's
  problem. Three consecutive post-eighth-iteration changes (cosine
  annealing, attention gates at two bottleneck widths) have now each
  underperformed the eighth iteration's plain regularized UNet, which
  remains the best-performing configuration found. Full writeup in
  `docs/real_data_validation.md`.

- **Toggleable attention gates**: `ResAttentionUnetConfig` gains
  `use_attention: bool = True` (default preserves every existing
  config/call site's behavior byte-for-byte) -- `False` builds the same
  `ResAttentionUNet` class and forward pass with the attention gates
  removed entirely, isolating whether the attention mechanism itself,
  versus the residual-block architecture it sits on, is responsible for
  a given result.
- Twelfth ACDC validation iteration (2026-08-31): `examples/
  validate_acdc.py` sets `use_attention=False`, motivated by two
  consecutive attention-gate configurations (tenth, eleventh) both
  underperforming the eighth iteration's baseline -- this iteration
  asks whether attention itself, or the residual architecture
  underneath it, was responsible. Otherwise identical to the tenth
  iteration (the "standard" `attention_reduction=2` run), isolating
  attention on/off as the sole variable. Validation Dice reached
  **0.8278 at epoch 15**, the best of any iteration so far, then
  training showed the same kind of late-training collapse the eleventh
  iteration had (val Dice 0.81 -> 0.55 at epoch 23) -- despite this run
  having zero attention gates, ruling out attention as that
  instability's cause. Early-stopped at epoch 25 with its best
  checkpoint from epoch 15. Result: the hypothesis did not hold --
  macro test Dice fell further still (0.7313 -> **0.7200**, now the
  worst multi-class result in this project), with damage spread across
  myocardium (Dice 0.7598 -> 0.6856, HD95 26.8mm -> 54.4mm) and LV
  (Dice 0.8610 -> 0.8288), not concentrated in RV as the tenth
  iteration's was. Four consecutive post-eighth-iteration changes
  (cosine annealing, attention at two bottleneck widths, no attention)
  have now each underperformed the eighth iteration's plain regularized
  UNet, which remains by a clear margin the best-performing
  configuration found. Full writeup in `docs/real_data_validation.md`.

- **ACDC results visualization**: `examples/validate_acdc.py` gains an
  optional `--visualize` flag (off by default, so every iteration's
  numbers above were produced without it) that runs
  `miai_pipeline.stages.visualization.VisualizationStage` -- the same
  class `examples/segmentation_pipeline.py`'s generic pipeline already
  uses, unmodified -- over every held-out test-set image right after
  evaluation, writing one QC slice-montage PNG per case. Also adds a
  new script, `examples/visualize_acdc_results.py`, which builds
  training-curve, ground-truth-vs-prediction, per-iteration
  metric-summary, and points-of-interest (best/median/worst test-case
  Dice, ground truth vs. prediction vs. |difference|) plots from a
  completed run's logs and `evaluation_report.json` using
  `miai_visualization` (until now wired into the generic pipeline demo
  only, never into this validation effort) -- no retraining needed.
  The comparison plots, per-class Dice box plot, QC montages, and
  points-of-interest grid are all produced for both the twelfth
  iteration and the eighth-iteration baseline on the same test cases,
  since both share the same `seed=42` patient-level test split. Full
  description in `docs/real_data_validation.md`'s "Visualizing
  results" section.

- **Class-weighted Dice loss**: `miai_segmentation.three_d.train.
  TrainingConfig` (shared by `two_d`) gains a `class_weights:
  tuple[float, ...] | None = None` field (default `None`, so every
  existing config keeps `DiceLoss`'s own unweighted behavior
  byte-for-byte), wired straight through to `DiceLoss`'s own `weight`
  argument. Its length is validated against the loss's actual channel
  count (`num_classes` in multi-class mode, `1` in binary mode) before
  training starts, raising `SegmentationError` immediately on a
  mismatch.
- Thirteenth ACDC validation iteration (2026-09-01): after four
  consecutive architecture-side changes (cosine annealing, attention
  at two bottleneck widths, no attention) each underperformed the
  eighth iteration's plain regularized `UNet`, this iteration reverts
  `examples/validate_acdc.py` to that exact baseline architecture and
  changes a single new lever instead -- `class_weights=(0.5, 2.0, 1.5,
  1.0)` for (background, RV, myocardium, LV), motivated by RV and
  myocardium being the consistently weakest structures since the
  seventh iteration's per-class breakdown. Training ran the full
  50-epoch budget without early stopping firing, reaching a new
  project-best validation Dice of **0.8378 at epoch 44**. Result: the
  hypothesis did not hold -- macro test Dice fell from the eighth
  iteration's 0.7740 to **0.7348**, with RV essentially flat (0.6952
  -> 0.6902) despite the highest weight, myocardium slightly worse
  (0.7593 -> 0.7280) despite also being up-weighted, and LV -- left at
  weight 1.0 -- taking the largest hit of the three (0.8676 -> 0.7861).
  Still well above the twelfth iteration's 0.7200 floor, but a useful
  negative result: simple loss reweighting, applied directly, did not
  close the RV/Myo gap for free. The eighth iteration's plain,
  unweighted `UNet` remains the best-performing configuration found.
  Full writeup in `docs/real_data_validation.md`.

## [1.0.0] - 2026-08-28

First `1.0.0` release: the ACDC real-data validation effort (five
iterations, 2026-08-26/27/28) is the milestone that closes out MIAI's
`0.x` "shake out the API shape" phase (`docs/compatibility_policy.md`)
-- every segmentation modality (`three_d`, `two_d`, `two_half_d`,
binary and now multi-class) has been exercised end to end against
real clinical data, not just synthetic test fixtures, and the public
API has stabilized enough to commit to standard SemVer guarantees
going forward (see `docs/compatibility_policy.md`'s "From 1.0
onward"). No breaking changes in this release -- every addition below
is backward compatible with `0.16.0`.

### Added

- **Multi-class segmentation support** (2026-08-28): `miai_segmentation`
  and `miai_evaluation` gain a genuine multi-class path, not just a
  binary-only one. `miai_segmentation.three_d.train.TrainingConfig`,
  `miai_segmentation.{two_d,three_d}.infer.InferenceConfig`, and
  `miai_evaluation.metrics.MetricsConfig` each gain a `num_classes`
  field (default `1`, preserving every existing binary caller's
  behavior byte-for-byte). Setting `num_classes > 1` switches training
  to softmax logits and `DiceLoss(softmax=True, to_onehot_y=True)`
  (in place of sigmoid + threshold), inference to argmax (in place of
  a probability threshold), and evaluation to one-hot-encoded,
  background-excluded (`include_background=False`) metrics, plus a new
  per-class Dice breakdown (`dice_class_1`, `dice_class_2`, ...) so a
  single macro-averaged Dice can't hide which class a model struggles
  with. Fully backward compatible -- `num_classes=1` is the default
  everywhere, so no existing config or call site changes behavior.
- Fifth ACDC validation iteration (2026-08-28): `examples/validate_acdc.py`
  puts the new multi-class support to real use -- ACDC's ground truth
  (background, right ventricle, myocardium, left ventricle) is no
  longer merged into one "whole heart" binary label; the model now
  trains and is scored on all 4 classes directly (`_NUM_CLASSES = 4`
  wired into `TrainingConfig`/`InferenceConfig`/`MetricsConfig`; the
  label preparation step, formerly `_binarize_label`, now just casts
  ACDC's already-4-class ground truth to `uint8`). Same full
  150-patient/300-case dataset, 2D per-slice UNet architecture
  (`out_channels=4`), patient-level 180/60/60 split, augmentation, and
  25-epoch budget as the fourth iteration. Result: macro (foreground-
  only) mean test Dice **0.72**, with a per-class breakdown showing the
  difficulty is concentrated in the right ventricle (RV Dice 0.58, a
  thin crescent-shaped structure) more than the myocardium (0.72) or
  left ventricle (0.86, close to the fourth iteration's binary
  result) -- a well-known pattern in cardiac segmentation, and a more
  clinically meaningful result than a single binary "whole heart"
  number. Full writeup in `docs/real_data_validation.md`.
- Fourth ACDC validation iteration (2026-08-27): `examples/validate_acdc.py`
  scales `DEFAULT_PATIENTS` from a 50-patient subset (100 cases) to the
  full ACDC dataset -- `patient001` through `patient150` (300 cases,
  both ED and ES), covering the official training split (001-100, 5
  pathology groups) and testing split (101-150). Same 2D per-slice
  architecture, patient-level split, and augmentation as the third
  iteration; epoch budget reduced 40 -> 25 given ~3x the per-epoch
  example count (patient-level split now 180/60/60 cases from
  90/30/30 patients). Result: mean test Dice rose from 0.71 (50
  patients) to **0.82** (150 patients), with specificity 0.996 and
  sensitivity 0.81 -- confirming that more training data still helps
  substantially once the model's inductive bias (2D per-slice) is
  correctly matched to the data, unlike the second iteration's
  combined levers on top of a mismatched 3D architecture, which
  bought nothing. Full writeup in `docs/real_data_validation.md`.
- Third ACDC validation iteration (2026-08-27): `examples/validate_acdc.py`
  switches from a 3D UNet to MIAI's per-slice 2D UNet
  (`architecture.modality = "two_d"`, already wired into every pipeline
  stage since Phase 8) -- prompted by the observation that ACDC's
  cine-MRI is acquired as a stack of independent 2D short-axis slices
  (in-plane ~1.5-2mm, through-plane ~6-10mm, only 6-15 slices per case),
  not a true volumetric scan, so a 3D UNet imposes a spatial
  relationship between slices the acquisition never has. No new data or
  staging needed: `expand_to_slice_dicts` turns each already-on-disk
  ED/ES volume into one training example per slice (~700+ 2D examples
  from the same 60 training volumes), and
  `miai_segmentation.two_d.infer.run_case_inference` reassembles slice
  predictions back into one volume per case for evaluation. Same
  architecture depth, patient-level split, augmentation, and epoch
  budget as the second iteration -- only the modality changed. Result:
  mean test Dice jumped from ~0.08 (both prior 3D iterations) to
  **0.71**, with specificity reaching 0.99 (versus ~0.5-0.6 before) --
  the first ACDC iteration where the model actually generalizes, not
  just validates the pipeline's wiring. Confirms matching the model's
  inductive bias to the data's real acquisition geometry mattered far
  more than the second iteration's combined data/augmentation/capacity
  levers. Full writeup in `docs/real_data_validation.md`.
- Second ACDC validation iteration (2026-08-27): `examples/validate_acdc.py`
  now uses both ED and ES frames per patient (50 patients, up to 100
  cases, up from 30 ED-only), a **patient-level** train/val/test split
  (a patient's ED and ES frames always land in the same split -- the
  previous `DatasetStage`-based case-level split couldn't guarantee
  that once both frames were in play), stronger augmentation (random
  90-degree rotation and intensity shift added to the existing random
  flip), a deeper 3-level UNet, finer resampling spacing, and more
  epochs (60, up from 40) -- every lever from the first run's
  improvement list pulled at once, except multi-class labels. Also
  found and fixed a real bug along the way: the inference stage's
  sliding-window `roi_size` was sized for the first run's shallower
  volumes and silently under-covered the taller, finer-spaced volumes
  this run produced, cutting mean test Dice roughly in half versus
  what the same checkpoint scores once `roi_size` covers a full case in
  one window (0.040 -> 0.084). Honest result even with the fix and
  every lever combined: mean test Dice 0.082, statistically
  indistinguishable from the first run's 0.088 despite 3.3x the data,
  the leak-proof split, and the bigger model -- val Dice reaching 0.72
  shows the architecture has capacity, so the bottleneck is training
  data volume, not model size. Full before/after writeup in
  `docs/real_data_validation.md`.
- First real-data validation (2026-08-26): `examples/validate_acdc.py`
  runs the full `miai_pipeline` (preprocess -> split -> train ->
  sliding-window inference -> evaluate) end to end against the public
  ACDC cardiac cine-MRI dataset, not just synthetic volumes. Scoped to
  binary "whole heart" segmentation (multi-class support would be a
  separate feature addition -- `miai_segmentation` is currently
  binary-only), one ED frame per patient (avoids patient-level split
  leakage), and a 30-patient subset. Found and fixed two real gaps
  synthetic data never exercised: independently-resampled image/label
  geometry can round to different sizes by a voxel (worked around in
  the script via resampling labels directly onto their preprocessed
  image's grid), and a 3D UNet needs its input padded to a multiple of
  its total downsampling stride, which nothing enforced for
  arbitrarily-sized real volumes -- `miai_transforms.compose
  .TRANSFORM_REGISTRY` gained a general-purpose `"divisible_pad"` entry
  (`monai.transforms.DivisiblePadd`) for this, with its own test.
  Full writeup, including the honest result (pipeline validated sound;
  the toy-scale 18-patient training run itself overfits -- val Dice
  0.83, held-out test Dice 0.09, as expected at this data scale) in the
  new `docs/real_data_validation.md`.
- Full-repository test coverage reaches 100% (2026-08-26), closing the
  remaining gaps from the 2026-08-26 audit: deterministic error/branch
  tests added for `miai_core.io` (malformed YAML, non-mapping JSON,
  unserializable YAML/JSON writes), `miai_datasets.slices` (zero-depth
  volume guard, via mocking `_read_depth` since SimpleITK cannot write a
  genuine zero-depth NIfTI to disk), `miai_dicom.series` (files without
  a `SeriesInstanceUID`, sort-key fallback through
  `ImagePositionPatient` and through the final default when neither
  `InstanceNumber` nor position is present), `miai_dicom.metadata`
  (`_to_jsonable`'s generic `str()` fallback for non-numeric/date
  types), `miai_dicom.io` (`write_dicom` wrapping pydicom's `save_as`
  failure for a dataset without `file_meta`), `miai_evaluation.metrics`
  (`_volume_similarity`'s both-empty-masks edge case),
  `miai_pipeline.stages.preprocessing`
  (`PreprocessingStage._normalize`'s unknown-normalization guard, via
  `model_construct` to bypass the `Literal` type's validation),
  `miai_pipeline.stages.training` (2D/2.5D modality now also expands
  the validation split to slices, not just the training split),
  `miai_pipeline.stages.dicom_to_nifti` (wrapping SimpleITK's
  `ImageSeriesReader` `RuntimeError` for an undecodable series), and
  `miai_segmentation.three_d.infer` (the non-tensor
  `sliding_window_inference` output guard, and the post-loop
  `source_paths`/`prediction_paths` length-mismatch check).
  `miai_foundation_models.extractor`'s `token_pooling="mean"` branch and
  `FeatureExtractor.from_pretrained` (mocking
  `transformers.AutoModel`/`AutoImageProcessor` via `patch.object` on
  the module resolved through `sys.modules`, since `huggingface.co` is
  unreachable in this sandbox and a plain dotted-string `patch()` trips
  `transformers`' lazy-loading backend check for the missing
  `torchvision` optional dependency) are also now covered.
  `miai_pipeline.cli`'s `if __name__ == "__main__":` guard and
  `miai_core.utils.set_seed`'s defensive `except ImportError` around an
  always-present `numpy` import are marked `# pragma: no cover` instead
  (untestable-by-definition and not worth mocking `sys.modules` for,
  respectively).
- Test coverage for previously-thin branches, surfaced by a project
  audit (2026-08-26): `tests/test_registration_register.py` now covers
  `miai_registration.register`'s `"affine"`/`"bspline"` transform types
  and `"mattes_mutual_information"`/`"correlation"` metrics (previously
  only `"rigid"`/`"mean_squares"` ran, plus the unknown-value error
  paths); `tests/test_visualization_curves.py`,
  `test_visualization_metrics.py`, and `test_visualization_slices.py`
  now cover the `title`-set branches, `plot_metric_summary`'s
  >6-label tick-rotation branch, and `plot_montage`'s zero-size-axis
  and leftover-panel (non-square slice count) branches. All four
  modules reach 100% coverage (previously 89%-98%).
- `tests/test_segmentation_two_d_infer.py`: dedicated unit tests for
  `miai_segmentation.two_d.infer` (previously only exercised indirectly
  through the pipeline stage/end-to-end tests), raising its coverage
  from 66% to 100%. Covers `run_inference`'s happy path plus its
  mismatched-`source_paths` branches (loader yields more or fewer items
  than expected), `run_case_inference`'s alignment checks
  (`case_slice_counts`/`source_paths` length mismatch, fewer slices
  than expected, leftover unconsumed slices) and its multi-case
  reassembly (`source_paths` order preserved, per-case depth correct),
  and `_predict_slice_mask`'s defensive non-tensor-model-output check.
- `miai_segmentation.two_d`: per-slice 2D segmentation architectures --
  `UNetConfig`/`build_unet` (`spatial_dims=2`) and
  `AttentionUnetConfig`/`build_attention_unet`
  (`monai.networks.nets.AttentionUnet`, Oktay et al. 2018), dispatched
  via `ArchitectureConfig`/`build_model`, following
  `miai_segmentation.three_d`'s pattern. `miai_segmentation.two_d.train`
  re-exports `three_d.train` unchanged (the training loop is
  dimension-agnostic); `miai_segmentation.two_d.infer` is a 2D-window
  (`roi_size: tuple[int, int]`) variant of `three_d.infer`.
- `miai_segmentation.two_half_d`: the 2.5D (stacked-adjacent-slice)
  architecture -- `StackedUNetConfig`/`build_stacked_unet`, a 2D UNet
  whose `in_channels` is the number of stacked adjacent slices,
  predicting the center slice's mask, dispatched via
  `ArchitectureConfig`/`build_model` for the same shape as the other two
  modalities even though only one architecture exists yet. Training
  re-exports `three_d.train`; inference re-exports `two_d.infer`
  (spatially identical -- both are a 2D sliding window; only the model's
  channel count differs, handled by the model itself).
- `tests/test_segmentation_reexports.py`: asserts the `two_d`/
  `two_half_d` re-export modules actually point at the objects they
  claim to (`train_model`/`TrainingConfig` from `three_d.train`,
  `run_inference`/`InferenceConfig` from `two_d.infer` for
  `two_half_d`), so a future refactor that shadows a re-export with a
  divergent definition fails loudly.
- `miai_segmentation.modality` (internal, not re-exported from
  `miai_segmentation`'s root `__init__.py`): `SegmentationModalityConfig`/
  `build_model_for_modality` and `SegmentationInferenceConfig`/
  `inference_config_for_modality`, which `miai_pipeline.stages.training`/
  `.inference`/`.export` now use to select a segmentation modality
  (`three_d`, `two_d`, or `two_half_d`) from one config field instead of
  hardcoding `three_d`.
- `miai_datasets.slices.expand_to_slice_dicts`: expands case-level data
  dicts into one dict per slice (reading each volume's depth from its
  file header only, via `SimpleITK.ImageFileReader`), bridging
  `miai_datasets.manifest.manifest_split_to_data_dicts`'s case-level
  output to what `two_d`/`two_half_d` need at the per-slice level.
- `miai_transforms.slice_transforms.ExtractSliced`/`ExtractSliceStackd`
  (registered in `TRANSFORM_REGISTRY` as `extract_slice`/
  `extract_slice_stack`): reduce an already-loaded `(C, D, H, W)` volume
  array to the 2D (`ExtractSliced`) or stacked-adjacent-slice
  (`ExtractSliceStackd`) input a slice-level model expects, indexed by a
  `"slice_index"` entry `expand_to_slice_dicts` adds to each data dict.
- `miai_segmentation.two_d.infer.run_case_inference` (also re-exported
  from `miai_segmentation.two_half_d.infer`): consumes a slice-level
  `DataLoader` (`case_slice_counts` items per case) and reassembles each
  case's per-slice predictions back into one `(D, H, W)` volume, so
  `InferenceStage`'s one-prediction-file-per-case contract is unchanged
  regardless of modality.
- `TrainingStage`/`InferenceStage`/`ExportStage`: `two_d` and
  `two_half_d` are now selectable via `architecture.modality` in
  pipeline YAML (previously only usable standalone) -- see
  `docs/user_guide.md`'s "2D and 2.5D segmentation" section for a full
  example, including the `extract_slice`/`extract_slice_stack`
  transforms those modalities need.
- `tests/test_pipeline_two_d_modality.py`: end-to-end integration test
  training and running inference with `architecture.modality: two_d`,
  confirming the reassembled prediction volume matches the source
  case's full depth/size.

### Changed

- `docs/roadmap.md`'s Phase 6 heading: `*(current)*` -> `*(complete)*`
  -- stale marker left over from before Phases 7-8 shipped.
- `README.md`'s "Installation" section: now leads with `pip install
  pymiai` (the first PyPI release, `v0.16.0`, is live), with the
  from-source instructions kept as a secondary "for development" path.
- `docs/release_process.md`: recorded that `pymiai` 0.16.0 published to
  PyPI successfully on 2026-08-18 -- TestPyPI dry run verified, then the
  real release approved and confirmed installable from pypi.org.
- `src/miai_segmentation/__init__.py`: docstring and `__version__`
  (`0.2.0` -> `0.3.0`) updated now that all three modalities are
  implemented; also corrects an inaccurate claim that
  `TrainingStage`/`InferenceStage` already select a modality from YAML
  -- they still hardcode `three_d` (see "Not done" below).
- `docs/roadmap.md`'s Phase 8 section: checked off `two_d`/`two_half_d`
  pipeline wiring; marked the phase **complete** (previously
  "architectures done, pipeline wiring pending").
- `src/miai_segmentation/__init__.py`: docstring and `__version__`
  (`0.3.0` -> `0.4.0`) updated -- all three modalities are now wired
  into the pipeline stages, not just usable standalone.
- **Breaking (pre-1.0, no deprecation cycle needed per
  `docs/compatibility_policy.md`):** `TrainingStageConfig.architecture`,
  `InferenceStageConfig.architecture`/`.inference`, and
  `ExportStageConfig.architecture` now take
  `miai_segmentation.modality.SegmentationModalityConfig`/
  `SegmentationInferenceConfig` instead of
  `miai_segmentation.three_d.models.ArchitectureConfig`/`.infer.
  InferenceConfig` directly. Existing YAML configs need
  `architecture: {kind: ..., unet: ...}` rewritten as
  `architecture: {modality: three_d, three_d: {kind: ..., unet: ...}}`,
  and `inference: {roi_size: ..., ...}` as
  `inference: {three_d: {roi_size: ..., ...}}` (see
  `examples/configs/pipeline.yaml` and `docs/user_guide.md` for the
  updated shape).
- `docs/architecture.md`, `docs/api_design.md`: corrected stale
  "`two_d`/`two_half_d` planned" language now that all three modalities
  are implemented and wired in.

## [0.16.0] - 2026-08-18

### Added

- `SECURITY.md`: vulnerability reporting policy ahead of making the repo
  public -- supported-versions table (only the latest `0.x.y` release,
  matching MIAI's pre-1.0 versioning), and instructions to use GitHub's
  private vulnerability reporting instead of a public issue.
- `CODE_OF_CONDUCT.md`: Contributor Covenant v2.1, GitHub's standard
  template, filled in for this repository.
- A non-clinical-use disclaimer in `README.md`'s status callout: MIAI is
  research/engineering tooling, not a regulatory-cleared medical device,
  and should not drive clinical decisions without independent validation.
- `miai_segmentation.three_d`: a new subpackage organizing MIAI's
  segmentation package by imaging modality (3D now, 2D and 2.5D planned
  -- see `docs/roadmap.md`'s Phase 8). Moved the existing UNet reference
  model, training loop, and sliding-window inference here unchanged, and
  added `SegResNetConfig`/`build_segresnet` (`monai.networks.nets.
  SegResNet`) as a second 3D architecture, plus `ArchitectureConfig`/
  `build_model` as a single dispatch point so callers pick an
  architecture via one `kind: "unet" | "segresnet"` field instead of
  calling a per-architecture builder directly.
- `docs/compatibility_policy.md`: defines what counts as MIAI's public
  API (scoped to `docs/api_design.md`'s "Package public surface"
  section), the pre-1.0 versioning stance (0.x allows breaking changes
  with only a MINOR bump, no deprecation cycle required), the SemVer
  rules from 1.0 onward, and the 4-step deprecation cycle that applies
  once 1.0 ships.
- `docs/user_guide.md`: task-oriented usage guide beyond the README --
  core concepts (`MIAIBaseConfig`, `PipelineStage`, `PipelineContext`),
  a full walkthrough of `examples/configs/pipeline.yaml` stage by stage,
  running the pipeline from Python and from the `miai-pipeline` CLI, a
  table of optional stages outside the main clinical workflow, a
  per-package quick-reference table for using one package standalone,
  and a troubleshooting Q&A section.
- `.github/workflows/publish.yml`: PyPI Trusted Publishing (OIDC)
  workflow -- builds sdist/wheel, and publishes to TestPyPI (manual
  dispatch only) or PyPI (on a published GitHub Release, or manual
  dispatch) via `pypa/gh-action-pypi-publish`. No PyPI API token is
  stored; publishing relies on this repository being registered as a
  trusted publisher on PyPI/TestPyPI.
- `docs/release_process.md`: the one-time PyPI/TestPyPI trusted-publisher
  setup (completed by the maintainer -- see below), the repeatable
  release-cutting procedure, and confirmation that a local `python -m
  build` was verified to build/install/import cleanly.

### Removed

- `CLAUDE.md` (Claude Code project instructions): dropped from the
  published repo and added to `.gitignore` ahead of making this repo
  public. Kept locally only; contained no secrets, just AI-assistant
  tooling instructions the maintainer didn't want in the public repo.

### Changed

- `miai_pipeline.stages.training.TrainingStageConfig`,
  `.inference.InferenceStageConfig`, and `.export.ExportStageConfig`:
  renamed the `unet: UNetConfig` field to `architecture:
  ArchitectureConfig`, so a pipeline YAML now selects the 3D
  architecture (and its settings) under `architecture.kind` /
  `architecture.unet` / `architecture.segresnet` instead of assuming
  UNet. Updated `examples/configs/pipeline.yaml` to match. This is a
  breaking config-shape change (`unet:` -> `architecture: {kind: unet,
  unet: {...}}`); acceptable pre-1.0 with no PyPI-published consumers.
- Moved `miai_segmentation.models`/`.train`/`.infer` to
  `miai_segmentation.three_d.models`/`.train`/`.infer`. Every internal
  caller (`miai_pipeline.stages.*`, tests) and the mypy per-module
  override lists in `pyproject.toml` were updated to the new paths.
- Refreshed `locks/requirements-lock.txt` via 7 of the 10 open Dependabot
  `pip` PRs: `ast-serialize`, `filelock`, `fsspec`, `hf-xet`, `librt`,
  `packaging`, `tqdm` (each landed at whatever their latest PyPI release was
  at regeneration time, which for `ast-serialize`/`filelock`/`librt` was
  newer than the specific version Dependabot's PR had originally proposed).
  The other 3 open PRs (`nvidia-cuda-cupti-13.3.75`, `nvidia-cufft-12.3.0.29`,
  `nvidia-nccl-cu13-2.30.7`) were intentionally **not** merged: torch 2.13.0
  pins these exact `nvidia-cu13` versions as hard `==` dependencies, so
  bumping them independently is not resolvable without also bumping torch.
  Those 3 PRs should be closed rather than merged.
- `docs/api_design.md`'s "Package public surface" section: rewritten to
  formally describe the hierarchical sub-namespace pattern already used
  by `miai_pipeline`/`.stages` and `miai_segmentation`/`.three_d` --
  public API is the union of the root `__init__.py` plus any documented
  sub-namespace `__init__.py` (one with its own module docstring and
  explicit `__all__`); a module not re-exported at either level is
  internal; sub-namespaces exist to avoid name collisions between
  mutually-exclusive variants of the same role, not for organizing by
  package size, and shouldn't be introduced preemptively.
- `src/miai_segmentation/__init__.py`: docstring extended with an
  explicit paragraph signposting the hierarchical pattern and pointing
  to `docs/api_design.md` for the rationale.
- `docs/coding_standards.md`'s "Versioning and commits" section: first
  bullet now links to `docs/compatibility_policy.md`.
- `README.md`: added `docs/user_guide.md` and
  `docs/compatibility_policy.md` to the documentation table, and a
  pointer to `docs/user_guide.md` ahead of the Installation section's
  quick-example code blocks.
- `pyproject.toml`'s PyPI package name: `miai-core` -> `miai`. The name
  `miai-core` would have collided in spirit with the `miai_core`
  utilities sub-package while actually installing all 14 `miai_*`
  packages (a monorepo/single-`pyproject.toml` side effect); `miai`
  avoids that confusion. `.github/workflows/publish.yml` and
  `docs/release_process.md` updated to match; no release has been
  published under either name yet, so this is not a breaking rename.
- `pyproject.toml`'s `Development Status` classifier: `2 - Pre-Alpha` ->
  `4 - Beta`. Reflects that all 14 packages are implemented and tested;
  not `5 - Production/Stable` since the project is still pre-1.0 (see
  `docs/compatibility_policy.md`) with no published release yet.
- `pyproject.toml`'s PyPI package name: `miai` -> `pymiai`. PyPI's
  pending-trusted-publisher form rejected `miai` as "too similar to an
  existing project" (a fuzzy/confusable-name check beyond exact PEP 503
  normalization, more aggressive for short names, without disclosing the
  colliding project); `pymiai` was accepted. `.github/workflows/publish.yml`
  and `README.md` updated to match. The maintainer has since completed
  the PyPI/TestPyPI trusted-publisher registration, created the GitHub
  `pypi`/`testpypi` environments, and made the repository public --
  `docs/release_process.md` rewritten to reflect this is all done; the
  only remaining step is cutting an actual release.

## [0.15.0] - 2026-08-17

### Added

- `pydocstyle` added to CI, deepening `interrogate`'s presence-only
  docstring check with a completeness check: well-formed summary lines
  (blank line between summary and extended description, summary ending
  in punctuation), no missing docstrings on magic methods (`__len__`,
  `__contains__`, etc.), and consistent Google-convention formatting.
  Configured under `[tool.pydocstyle]` in `pyproject.toml`, scoped to
  `src/` only (mirrors `interrogate`'s `tests`/`examples`/`scripts`
  exclusion). Fixed the ~40 real violations this surfaced across
  `miai_pipeline` (every concrete stage's `__init__`/`run` was
  undocumented -- the class docstring's Reads/Writes contract doesn't
  substitute for pydocstyle's per-method requirement), `miai_dicom`,
  `miai_core`, `miai_diffusion`, `miai_foundation_models`, and
  `miai_transforms`.

### Tests

- Added `test_end_to_end_reconstruction_feature_extraction_visualization`
  to `tests/test_pipeline_end_to_end.py`, chaining
  `dicom_to_nifti -> preprocessing -> {reconstruction, feature_extraction,
  visualization}` as one config-driven `Pipeline`. Previously these three
  optional stages were only covered in isolation, each starting from a
  hand-built context with `preprocessed_paths` injected directly by the
  test; this confirms all three consume the real output of a preceding
  `PreprocessingStage` instead.

## [0.14.0] - 2026-08-05

### Added

- `examples/configs/pipeline.yaml`: a real, runnable pipeline config for the
  full main clinical workflow (DICOM -> NIfTI -> Preprocessing -> Dataset ->
  Training -> Inference -> Evaluation), closing a dangling reference --
  `README.md`'s "Quick example" sections and the `miai-pipeline` CLI usage
  snippet already mentioned `configs/pipeline.yaml` as if it existed, but no
  such file was ever committed. Deliberately tiny architecture/epoch count
  (3-level UNet, 5 epochs) so it finishes in under a minute on CPU with
  synthetic data. First of three planned `miai-examples` deliverables (see
  `docs/roadmap.md`); the accompanying runnable script
  (`examples/segmentation_pipeline.py`) that generates the synthetic
  DICOM/label data this config expects comes next.
- `examples/output/` added to `.gitignore` (generated when the example
  scripts run).
- `examples/segmentation_pipeline.py`: a runnable script (standalone --
  does not import `tests/conftest.py`, which is test-only fixture code)
  that generates a small synthetic dataset (10 multi-slice DICOM series
  plus matching NIfTI labels, no patient data) and runs it through
  `examples/configs/pipeline.yaml` end to end via `Pipeline.from_config`,
  printing the dataset split, checkpoint path, prediction count, and mean
  evaluation metrics. Case directories are named `case_000`, `case_001`,
  ... so `miai_dicom.series.load_series`'s `sorted(rglob(...))` traversal
  discovers series in the same order labels are generated, keeping
  `DatasetStage`'s `label_context_key` alignment correct without needing
  to inspect series UIDs after the fact. Second of three planned
  `miai-examples` deliverables (see `docs/roadmap.md`).
- Adjusted `examples/configs/pipeline.yaml`'s `val_fraction`/`test_fraction`
  from 0.34/0.34 to 0.3/0.3, to pair with `segmentation_pipeline.py`'s 10
  synthetic cases for a cleaner 4/3/3 train/val/test split.
- `examples/diffusion_denoising.py`: a runnable script demonstrating an
  optional MIAI stage outside the main segmentation workflow -- trains a
  compact 3D DDPM on synthetic volumes, then simulates a "real noisy scan"
  by corrupting a known-clean volume with real forward-diffusion noise
  (`NoiseSchedule.q_sample`) and denoises it via reverse diffusion
  (`run_denoising`), reporting the MSE-to-clean before and after. Uses
  `miai_diffusion`'s package API directly (not a pipeline YAML config,
  unlike `segmentation_pipeline.py`), since diffusion training/denoising
  are optional stages meant to be used standalone.
- Rewrote `examples/README.md`, replacing the "not yet built" stub with a
  description of all three examples and how to run each. **This completes
  `miai-examples`** -- the last package in `docs/architecture.md`'s
  ecosystem diagram marked `[planned]`. Updated the ecosystem
  diagrams/status blurb in `README.md` and `docs/architecture.md`
  accordingly: all 14 planned MIAI packages are now implemented. PyPI
  packaging/publishing remains explicitly paused.

### Fixed

- Moved the dependency lockfile from `requirements-lock.txt` (repo root) to
  `locks/requirements-lock.txt`. Dependabot's `pip` ecosystem auto-discovered
  the root-level file as an independently manageable requirements file and
  opened one PR per outdated *transitive* package inside it -- including
  over a dozen `nvidia-*`/`triton` CUDA packages that PyPI's standard linux
  `torch` wheel pulls in even though CI only ever runs on CPU. Most of those
  PRs failed CI, since bumping a single pinned line breaks the internally
  consistent resolution the lockfile represents. `directory: "/"` in
  `.github/dependabot.yml` is non-recursive, so moving the file under
  `locks/` removes it from Dependabot's scan entirely; `ci.yml`,
  `security.yml`, and `docs/coding_standards.md` updated for the new path.

## [0.13.0] - 2026-07-24

### Added

- `mypy tests` (alongside the existing `mypy src`) in CI. Previously only
  production code was type-checked under `--strict`; the ~4000-line test
  suite had none. Fixed real issues this surfaced in `tests/conftest.py`:
  two pydicom UID assignments were widened to plain `str` through a
  reassigned variable (fixed by keeping the resolved value in a
  separate, correctly-typed `resolved_sop_instance_uid`, which also
  fixed a latent bug where `dataset.SOPInstanceUID` could end up `None`
  instead of the generated UID); `FileDataset(filename_or_obj=None, ...)`
  needed a documented `type: ignore[arg-type]` (pydicom's stub omits
  `None` even though the real implementation accepts it for in-memory
  datasets). Also added missing parameter/return type annotations across
  several test helpers, and a new `[[tool.mypy.overrides]]` block
  relaxing `disallow_untyped_calls` for the test modules that call
  SimpleITK directly (same rationale as the existing src-side override:
  SimpleITK ships no type stubs).
- `interrogate` docstring-completeness check in CI (`interrogate src`), gated
  by `[tool.interrogate]` in `pyproject.toml` (`style = "google"`,
  `fail-under = 85`, current actual is 87.1%). Enforces the *presence* of
  public function/class docstrings from `docs/api_design.md`'s Documentation
  contract -- not full Args/Returns/Raises completeness, which would need a
  stricter tool (e.g. `pydocstyle`'s `D417`) and a larger cleanup pass across
  all 13 packages to pass without a flood of pre-existing findings.
- `requirements-lock.txt`: a full dependency lockfile generated with
  `uv pip compile pyproject.toml --extra dev --universal --python-version 3.11`,
  pinning every direct and transitive dependency to an exact version.
  `ci.yml` and `security.yml` now install from it
  (`pip install -r requirements-lock.txt` + `pip install -e . --no-deps`)
  instead of resolving fresh against `pyproject.toml`'s lower bounds on every
  run, for fully reproducible installs. Regeneration instructions are in
  `docs/coding_standards.md`. Dependabot still targets `pyproject.toml`'s
  bounds; the lockfile needs a manual regen when those change (not yet
  automated).
- Four new metrics in `miai_evaluation.metrics.compute_case_metrics`, all
  opt-in via `MetricsConfig` (default `False`, so existing configs/reports
  are unaffected): `include_iou` (Jaccard index, via MONAI's `MeanIoU`),
  `include_sensitivity`/`include_specificity` (via MONAI's
  `ConfusionMatrixMetric`), and `include_volume_similarity` (Taha & Hanbury's
  `1 - |Vp - Vg| / (Vp + Vg)`, computed directly since MONAI doesn't expose
  this one -- a purely count-based measure that stays informative even when
  Dice/IoU are 0 because two same-sized masks don't overlap spatially).
- `--cov-fail-under=95` in pytest's `addopts`. Coverage was already reported
  (`--cov-report=term-missing`) but never enforced -- a PR could silently drop
  coverage and still pass CI. Current coverage is ~98% (2064 statements, 39
  missing per the last CI run), so 95% leaves headroom for legitimately
  hard-to-cover branches (error paths in torch/monai fix-forward code, mostly)
  while still catching a real regression.
- `cache: pip` in `actions/setup-python` for both `ci.yml` and `security.yml`, keyed
  on `pyproject.toml` -- the ~2 minute `pip install -e ".[dev]"` step (torch + monai
  are the bulk of it) now hits GitHub's Actions cache on unchanged dependencies
  instead of reinstalling from PyPI every run.
- `.github/dependabot.yml`: weekly automated dependency-update PRs for the `pip`
  (root `pyproject.toml`) and `github-actions` ecosystems.
- `.github/workflows/security.yml`: `pip-audit` scan against the fully installed
  environment, on a weekly schedule, on any `pyproject.toml` change, and on manual
  dispatch. Runs as its own workflow, separate from `lint-and-test`, so a newly
  disclosed vulnerability in a pinned dependency doesn't block unrelated PRs.

### Fixed

- `mypy tests` in CI (see below) surfaced several real over-tight production
  type signatures, fixed properly rather than papered over in tests:
  - `train_diffusion_model`, `train_model`, and `run_inference` required a
    concrete `monai.data.DataLoader`, even though each only ever iterates
    over it once per epoch. Loosened all three to
    `Iterable[dict[str, torch.Tensor]]`, letting tests pass plain lists or
    lightweight fakes without a type mismatch -- and documenting the real
    contract more accurately in the process.
  - `extract_embeddings_for_paths` required a concrete `FeatureExtractor`.
    Added a new `EmbeddingExtractor` Protocol (same pattern as the existing
    `_ImageProcessor` Protocol in the same module) exposing just
    `extract_volume_embedding`, and typed the parameter against that
    instead, so tests can inject a fake extractor without subclassing.
  - `plot_metric_summary`'s `values: dict[str, float | list[float]]` param
    rejected a plain `dict[str, float]` at call sites, since `dict` is
    invariant in its value type. Changed to `Mapping[str, float | list[float]]`
    (mypy's own suggested fix; the function never mutates `values`).
  - `evaluate_predictions` returned a bare `dict[str, object]`, so
    `report["mean"]["dice"]`-style indexing failed statically even though
    the real shape is fixed. Added an `EvaluationReport` TypedDict
    documenting the actual `{"per_case": [...], "mean": {...}}` structure.
- Pinned `setuptools>=83.0.0` in dev dependencies (PYSEC-2026-3447, found by the new
  pip-audit workflow on its first real run).
- Bumped `actions/checkout` to v5 and `actions/setup-python` to v6 in both workflows
  (the previous versions emit a Node.js 20 deprecation warning on GitHub's runners).

### Tests

- Deepened `tests/test_pipeline_end_to_end.py`, which previously only chained 3 of
  ~13 stages (`dicom_to_nifti -> preprocessing -> dataset`). Added: the same chain
  extended through `[registration]`; a full `dataset -> training -> inference ->
  evaluation` ML chain; and `diffusion_training -> denoising` /
  `training -> export` chains, which specifically verify that
  `model_checkpoint_path`/`diffusion_checkpoint_path` flow from one real stage to
  the next automatically -- no prior test exercised that handoff outside a
  hand-built context.
- Deepened `tests/test_segmentation_train.py` and `tests/test_diffusion_train.py`,
  which previously only checked that training ran without error and produced a
  loadable checkpoint ("smoke tests"). Added
  `test_train_model_actually_learns_to_segment`, which trains for 40 epochs on an
  easy synthetic pattern and asserts Dice is both above an absolute floor and
  higher than a freshly initialized model's, and
  `test_train_diffusion_model_actually_learns`, which trains for 60 epochs on a
  fixed learnable pattern and asserts the noise-prediction MSE on a fixed,
  seeded evaluation sample is lower than an untrained model's. Both compare
  against an untrained baseline of the same architecture rather than asserting
  an absolute threshold alone, so the tests fail if training silently becomes a
  no-op even if some residual accuracy would otherwise pass a bare threshold.

## [0.12.0] - 2026-07-16

### Added

- `py.typed` markers (PEP 561) added to all 13 implemented packages, confirmed to be
  included in a built wheel.
- `miai_pipeline.cli`: `miai-pipeline` console script (`run` / `validate` /
  `list-stages` subcommands), installed via a new `[project.scripts]` entry point.
- `PipelineContext.keys()`: read-only view of the keys currently set on a context,
  used by the CLI's `run` command to report what a pipeline produced.

### Fixed

- `examples/README.md` and `scripts/README.md` still read as if the project were in
  Phase 0/1; reworded to match the current (Phase 6) state.

## [0.11.0] - 2026-07-16

### Added

- `miai-visualization`: second Phase 6 package -- plotting tools for volumes,
  comparisons, training curves, metric summaries, and embeddings. Every plot is saved
  as a file (matplotlib's "Agg" backend, forced at package import), never shown
  interactively.
  - `miai_visualization.slices`: `plot_slice` (single slice, optional mask overlay) /
    `plot_montage` (grid of evenly-spaced slices).
  - `miai_visualization.comparison`: `plot_comparison` -- side-by-side images plus
    optional absolute-difference maps.
  - `miai_visualization.curves`: `plot_training_curves` -- lines from a CSV training log.
  - `miai_visualization.metrics`: `plot_metric_summary` -- bar/box plot of a per-case
    metric (visualizes values computed elsewhere, e.g. by `miai_evaluation` or
    `miai_reconstruction`; does not compute metrics itself).
  - `miai_visualization.embeddings`: `plot_embedding_projection` -- 2-component PCA
    (via `torch.linalg.svd`, no new dependency for this) scatter plot, e.g. of
    `miai_foundation_models` embeddings.
  - `miai_pipeline.stages.visualization.VisualizationStage`: optional pipeline stage
    writing a QC slice montage per case, `qc_visualization_paths`.
  - New dependency: `matplotlib`.

## [0.10.0] - 2026-07-16

### Added

- `miai-reconstruction`: first Phase 6 package -- MRI reconstruction from (simulated)
  k-space.
  - `miai_reconstruction.kspace`: `KSpaceReconstructionConfig` +
    `simulate_kspace`/`reconstruct_from_kspace` -- forward/inverse FFT via `torch.fft`
    (no new dependency for the core algorithm); `UndersamplingConfig` +
    `build_undersampling_mask`/`apply_undersampling` -- zero-filled reconstruction from
    an undersampled acquisition, with a fully-sampled k-space center (fastMRI-style mask).
  - `miai_reconstruction.metrics`: `reconstruction_quality` -- PSNR/SSIM via
    scikit-image, for photometric reconstruction-quality comparisons (distinct from
    `miai_evaluation`'s segmentation-mask metrics).
  - `miai_reconstruction.run`: `run_kspace_reconstruction` -- SimpleITK-only I/O,
    simulates k-space from an existing volume and reconstructs it.
  - `miai_pipeline.stages.reconstruction.ReconstructionStage`: optional pipeline stage,
    writing `reconstructed_paths`.
  - New dependency: `scikit-image`.
- Phase 6 ("Further ecosystem packages") begun.

## [0.9.0] - 2026-07-15

### Added

- `miai-deploy`: fourth and final Phase 5 package -- portable export and bundling of
  trained models. Reference task is portable export (TorchScript/ONNX artifact), not
  live model serving.
  - `miai_deploy.export`: `ExportConfig` + `export_model` -- loads a checkpoint into a
    given model, traces it (`torch.jit.trace`) or exports it (`torch.onnx.export`) to a
    portable inference artifact.
  - `miai_deploy.bundle`: `BundleMetadata` + `write_bundle` -- exports a model and writes
    it alongside a `metadata.yaml` (name/version/description) as a single bundle
    directory, so an exported model is never handed off without knowing what produced it.
  - `miai_pipeline.stages.export.ExportStage`: optional pipeline stage exporting the
    segmentation model trained by `TrainingStage`, writing `deploy_bundle_path`.
  - New dependency: `onnx`.
- Phase 5 ("Advanced modules": Registration, Diffusion, Foundation models, Deployment)
  is now complete.

## [0.8.0] - 2026-07-15

### Added

- `miai-foundation-models`: third Phase 5 package -- per-volume embeddings from a
  pretrained Hugging Face Hub vision model, for downstream retrieval/clustering/simple
  classification without fine-tuning.
  - `miai_foundation_models.extractor`: `FeatureExtractorConfig` + `FeatureExtractor` --
    wraps a 2D vision model (default `facebook/dinov2-small`, swappable via `model_id`)
    as a per-volume embedder using a slice-and-aggregate ("2.5D") strategy: every slice
    along `slice_axis` is embedded independently (CLS or mean token pooling), then the
    per-slice embeddings are pooled across slices (mean or max) into one vector.
    `FeatureExtractor.from_pretrained` downloads the model/processor from the Hub;
    the constructor also accepts an already-loaded model/processor pair.
  - `miai_foundation_models.run`: `extract_embeddings_for_paths` -- reads volumes via
    SimpleITK (same I/O convention as the rest of MIAI), extracts an embedding per case,
    and saves each as a `.pt` tensor file.
  - `miai_pipeline.stages.feature_extraction.FeatureExtractionStage`: optional pipeline
    stage, outside the main segmentation workflow, writing `embedding_paths`.
  - New dependencies: `transformers`, `huggingface_hub`, `Pillow`.

## [0.7.0] - 2026-07-15

### Added

- `miai-diffusion`: second Phase 5 package -- a from-scratch DDPM for
  volume denoising, on PyTorch only (no MONAI generative extension).
  - `miai_diffusion.schedule`: `NoiseScheduleConfig` + `NoiseSchedule`
    -- linear or cosine beta schedule, forward noising (`q_sample`)
    and the reverse denoising step (`p_sample_step`), following Ho,
    Jain & Abbeel, "Denoising Diffusion Probabilistic Models" (2020).
  - `miai_diffusion.model`: `DiffusionUNetConfig` + `DiffusionUNet` --
    a compact 3D UNet with sinusoidal timestep conditioning, built from
    plain `torch.nn` layers.
  - `miai_diffusion.train`: `DiffusionTrainingConfig` +
    `train_diffusion_model` -- the standard DDPM noise-prediction
    training objective (MSE between predicted and actual noise).
  - `miai_diffusion.denoise`: `DenoiseConfig` + `denoise_volume` /
    `run_denoising` -- denoises a real noisy volume by treating it as
    the schedule's `x_t` at a configurable `start_timestep` and running
    the reverse process down to `t=0` (an SDEdit-style use of a
    diffusion model as a restoration prior, not just for unconditional
    sampling from pure noise). File I/O via SimpleITK, consistent with
    the rest of MIAI.
  - `miai_pipeline.stages.diffusion_training.DiffusionTrainingStage`
    and `.stages.denoising.DenoisingStage`: new optional pipeline
    stages (unconditional training on `manifest["train"]`; denoising
    any list of volumes), separate from the main segmentation workflow.
  - No new third-party dependencies -- torch has been a MIAI dependency
    since Phase 4.

## [0.6.0] - 2026-07-15

### Added

- `miai-registration`: first package of Phase 5, image registration on
  SimpleITK.
  - `miai_registration.register`: `RegistrationConfig` +
    `register_images` -- rigid, affine, or bspline registration via
    `SimpleITK.ImageRegistrationMethod`, multi-resolution gradient
    descent, configurable metric (Mattes mutual information by
    default, or mean squares / correlation).
  - `miai_registration.apply`: `apply_transform` -- resamples another
    image (e.g. a label mask) onto the fixed grid with a previously
    estimated transform, using nearest-neighbor interpolation by
    default so label values are not corrupted.
  - `miai_registration.transform_io`: `write_transform` /
    `read_transform`.
  - `miai_pipeline.stages.registration.RegistrationStage`: new
    optional pipeline stage, registering every case onto a common
    fixed reference image (e.g. an atlas/template) -- fits between
    `preprocessing` and `dataset` for population-study-style
    workflows. Propagates each case's transform to its label too, if
    `label_context_key` is configured.
  - No new third-party dependencies -- SimpleITK has been a MIAI
    dependency since Phase 2.

## [0.5.0] - 2026-07-15

### Added

- `miai-evaluation`: closes out the `evaluation` stage left open since
  Phase 3.
  - `miai_evaluation.metrics`: `MetricsConfig` and `compute_case_metrics`
    -- Dice similarity coefficient and Hausdorff distance (95th
    percentile / "HD95" by default), via `monai.metrics`.
  - `miai_evaluation.evaluate`: `evaluate_predictions`, a file-based
    runner that reads prediction/ground-truth NIfTI pairs via
    SimpleITK (consistent with the rest of MIAI's image I/O),
    aggregates per-case metrics into a summary report, and optionally
    writes it as JSON.
  - `miai_pipeline.stages.evaluation.EvaluationStage`: concrete
    implementation (previously `NotImplementedError`), reading
    `prediction_paths` and the `manifest["test"]` ground truth labels
    (requires the manifest to have been built with
    `DatasetStage`'s `label_context_key`).
  - All three pipeline stages (`training`, `inference`, `evaluation`)
    are now concrete -- the clinical workflow interface defined in
    Phase 3 is fully implemented end to end.

## [0.4.0] - 2026-07-14

### Added

- Phase 4: MONAI integration -- reference binary 3D segmentation.
  - `miai_transforms`: `TransformConfig` / `TransformSpec` and
    `build_transforms`, a named registry (`TRANSFORM_REGISTRY`) of
    MONAI dictionary-based transforms composed from YAML.
  - `miai_datasets`: `manifest_split_to_data_dicts` (normalizes a
    `miai_pipeline` manifest split into MONAI data dicts), plus
    `build_dataset` / `build_dataloader` wrapping
    `monai.data.Dataset` / `CacheDataset` / `DataLoader`.
  - `miai_transforms.sitk_transforms.LoadImageSitkd`: MIAI's own
    SimpleITK-backed image loader, used in place of MONAI's
    `LoadImaged` (which needs an extra reader backend such as
    nibabel or itk). Keeps every array in SimpleITK's `(D, H, W)`
    axis convention end to end, so `run_inference` needs no axis
    transposition when writing predictions back out.
  - `miai_segmentation`: `build_unet` (MONAI `UNet`), `train_model`
    (Dice loss + Adam + Dice metric training loop with best-checkpoint
    saving), and `run_inference` (sliding-window inference, writing
    predictions as NIfTI with the source case's spatial metadata).
  - `miai_pipeline.stages.training.TrainingStage` and
    `.stages.inference.InferenceStage`: concrete implementations
    replacing the Phase 3 interfaces, wired to the three packages
    above. `EvaluationStage` remains an interface (lands with
    `miai-evaluation`).
  - `miai_pipeline.stages.dataset.DatasetStage`: new optional
    `label_context_key` config field, producing
    `{"image": ..., "label": ...}` manifest entries for supervised
    tasks instead of plain path strings (backward compatible: omitting
    it keeps the Phase 3 behavior).
  - Adds `torch` and `monai` as dependencies (SimpleITK, already a
    dependency since Phase 2, remains MIAI's only image I/O library).

## [0.3.0] - 2026-07-07

### Added

- Phase 3: `miai-pipeline` package implementation.
  - `miai_pipeline.context`: `PipelineContext`, a key-value store passed
    between stages.
  - `miai_pipeline.stage`: `PipelineStage`, the abstract base every
    stage implements (`name`, `config_cls`, `run`).
  - `miai_pipeline.pipeline`: `Pipeline`, which runs an ordered list of
    stages; `Pipeline.from_config` builds one from a YAML-loadable
    `PipelineConfig`, keyed by a stage type registry.
  - `miai_pipeline.stages.dicom_to_nifti`: `DicomToNiftiStage` — converts
    every DICOM series under a directory into a `.nii.gz` volume, using
    SimpleITK's `ImageSeriesReader` on top of `miai_dicom.series`.
  - `miai_pipeline.stages.preprocessing`: `PreprocessingStage` —
    resamples to a target voxel spacing and normalizes intensity
    (z-score, min-max, or none).
  - `miai_pipeline.stages.dataset`: `DatasetStage` — splits cases into
    train/val/test and writes a JSON manifest, reproducibly (seeded).
  - `miai_pipeline.stages.training` / `.inference` / `.evaluation`:
    abstract stage interfaces defining the contract for Phase 4
    (concrete implementations land once MONAI is integrated).
  - `miai_pipeline.exceptions`: `PipelineError`, `StageError`,
    `UnknownStageError`.
  - Test suite (28 tests, including a synthetic-DICOM end-to-end run
    through the full DICOM → NIfTI → preprocessing → dataset chain,
    both stage-by-stage and via a config file). Total repo test count
    is now 89. black, ruff, and mypy all pass.
  - Adds `SimpleITK` and `numpy` as dependencies.

## [0.2.0] - 2026-07-07

### Added

- Phase 2: `miai-dicom` package implementation.
  - `miai_dicom.io`: `read_dicom` / `write_dicom` / `is_dicom_file`,
    wrapping pydicom with MIAI's exception hierarchy.
  - `miai_dicom.metadata`: `extract_metadata` for a flat,
    JSON-serializable dictionary of core DICOM tags.
  - `miai_dicom.anonymize`: `anonymize`, a practical subset of the
    DICOM PS3.15 Basic Application Level Confidentiality Profile
    (removes direct identifiers, regenerates UIDs, flags
    `PatientIdentityRemoved`).
  - `miai_dicom.series`: `load_series` / `DicomSeries`, grouping a
    directory of DICOM files by `SeriesInstanceUID` and sorting each
    series into acquisition order.
  - `miai_dicom.validation`: `validate_dataset` / `is_valid_dataset`
    for checking a parsed dataset carries the tags a workflow needs.
  - `miai_dicom.exceptions`: `InvalidDicomFileError`.
  - Test suite (31 tests) using synthetic in-memory DICOM fixtures;
    total repo test count is now 61. black, ruff, and mypy all pass.

## [0.1.0] - 2026-07-06

### Added

- Phase 1: `miai-core` package implementation.
  - `miai_core.config`: `MIAIBaseConfig`, a Pydantic base class with
    `from_yaml` / `to_yaml` for reproducible, validated experiment
    configuration.
  - `miai_core.logging`: `configure_logging` / `get_logger` for consistent
    logging across all MIAI packages.
  - `miai_core.io`: YAML/JSON read/write helpers and `ensure_dir`, raising
    MIAI-specific exceptions instead of raw stdlib errors.
  - `miai_core.exceptions`: `MIAIError` hierarchy (`ConfigError`,
    `MIAIIOError`, `ValidationError`, `NotFoundError`).
  - `miai_core.typing`: shared `StrPath` / `JSONDict` aliases.
  - `miai_core.utils`: `set_seed`, `utc_timestamp`, `deep_update`.
  - Full test suite (30 tests) covering all modules.

## [0.0.1] - 2026-07-06

### Added

- Phase 0 project scaffold: repository structure, documentation set
  (vision, architecture, roadmap, coding standards, API design),
  contributing guide, MIT license, `pyproject.toml`, and CI configuration.
