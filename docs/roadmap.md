# Roadmap

MIAI is built step by step, one phase at a time. Each phase should be
functional and tested before moving to the next.

## Phase 0 — Project Design

- [x] Vision and mission
- [x] Ecosystem architecture
- [x] Documentation structure
- [x] Coding standards
- [x] API design principles
- [x] Repository scaffold (this commit)

## Phase 1 — `miai-core`

Common utilities every other package depends on:

- [x] Configuration system (YAML-based, Pydantic-validated)
- [x] Logging
- [x] IO utilities
- [x] Exceptions hierarchy
- [x] Typing utilities
- [x] General utilities
- [x] Test suite and CI

## Phase 2 — `miai-dicom`

- [x] Reading DICOM
- [x] Writing DICOM
- [x] Metadata extraction
- [x] Anonymization
- [x] Series loading
- [x] Validation

## Phase 3 — `miai-pipeline`

Clinical workflow orchestration:

```
DICOM → NIfTI → Preprocessing → Dataset → Training → Inference → Evaluation
```

- [x] Config-driven `Pipeline` / `PipelineStage` orchestration framework
- [x] `dicom_to_nifti` stage (DICOM series → NIfTI, via SimpleITK)
- [x] `preprocessing` stage (resampling, intensity normalization)
- [x] `dataset` stage (manifest generation, train/val/test split)
- [x] `training` stage (interface defined in Phase 3; concrete implementation in Phase 4)
- [x] `inference` stage (interface defined in Phase 3; concrete implementation in Phase 4)
- [x] `evaluation` stage (concrete implementation alongside `miai-evaluation`)

## Phase 4 — Integration with MONAI

- [x] `miai-transforms`: config-driven MONAI transform pipelines
- [x] `miai-datasets`: manifest -> MONAI `Dataset`/`CacheDataset`/`DataLoader`
- [x] `miai-segmentation`: reference binary 3D segmentation (MONAI `UNet`,
  Dice loss/metric training loop, sliding-window inference)
- [x] `miai_pipeline.stages.training.TrainingStage` / `.inference.InferenceStage`
  concrete implementations, wired to the three packages above
- [x] `dataset` stage: optional `label_context_key` for supervised manifests

## `miai-evaluation` — Dice / Hausdorff distance metrics

Closes out the `evaluation` stage left open in Phase 3/4:

- [x] `miai_evaluation.metrics`: Dice similarity coefficient, Hausdorff distance (HD95 by default)
- [x] `miai_evaluation.evaluate`: file-based `evaluate_predictions`, reading prediction/ground-truth
  NIfTI pairs via SimpleITK and aggregating per-case metrics into a summary report
- [x] `miai_pipeline.stages.evaluation.EvaluationStage`: concrete implementation, replacing the
  Phase 3 interface

## Phase 5 — Advanced modules

- [x] Registration
  - [x] `miai_registration.register`: `register_images` -- rigid/affine/bspline registration via
    `SimpleITK.ImageRegistrationMethod` (multi-resolution gradient descent)
  - [x] `miai_registration.apply`: `apply_transform` -- propagates an estimated transform to a
    paired image (e.g. a label mask, via nearest-neighbor interpolation)
  - [x] `miai_registration.transform_io`: save/load transforms
  - [x] `miai_pipeline.stages.registration.RegistrationStage`: optional pipeline stage, aligning
    every case onto a common fixed reference image (e.g. an atlas) between `preprocessing` and
    `dataset`
- [x] Diffusion
  - [x] `miai_diffusion.schedule`: `NoiseSchedule` -- forward (`q_sample`) and reverse
    (`p_sample_step`) DDPM process (Ho, Jain & Abbeel 2020), linear or cosine beta schedule
  - [x] `miai_diffusion.model`: `DiffusionUNet` -- compact 3D UNet with sinusoidal timestep
    conditioning, implemented from scratch in PyTorch (no MONAI generative extension)
  - [x] `miai_diffusion.train`: `train_diffusion_model` -- standard DDPM noise-prediction
    training loop
  - [x] `miai_diffusion.denoise`: `denoise_volume` / `run_denoising` -- SDEdit-style denoising
    of real noisy volumes via partial reverse diffusion from a configurable start timestep
  - [x] `miai_pipeline.stages.diffusion_training.DiffusionTrainingStage` /
    `.denoising.DenoisingStage`: optional pipeline stages, outside the main segmentation
    workflow
- [x] Foundation models
  - [x] `miai_foundation_models.extractor`: `FeatureExtractor` -- embeds a volume with a
    pretrained Hugging Face Hub vision model (default `facebook/dinov2-small`) using a
    slice-and-aggregate ("2.5D") strategy: run the 2D model over every slice, pool tokens
    per slice (CLS or mean), then pool across slices (mean or max) into one embedding
  - [x] `miai_foundation_models.run`: `extract_embeddings_for_paths` -- extracts and saves
    embeddings for a list of volumes on disk, SimpleITK-only I/O
  - [x] `miai_pipeline.stages.feature_extraction.FeatureExtractionStage`: optional pipeline
    stage, outside the main segmentation workflow, for embedding-based downstream tasks
    (retrieval, clustering, simple classification) without fine-tuning
- [x] Deployment
  - [x] `miai_deploy.export`: `export_model` -- exports a trained model to a portable
    inference format, TorchScript (`torch.jit.trace`, no extra dependency) or ONNX
    (`torch.onnx.export`); reference task is portable export, not live model serving
  - [x] `miai_deploy.bundle`: `BundleMetadata` + `write_bundle` -- packages an exported
    model together with reproducibility metadata (name/version/description) as a single
    bundle directory
  - [x] `miai_pipeline.stages.export.ExportStage`: optional pipeline stage exporting the
    trained segmentation model produced by `TrainingStage`, writing `deploy_bundle_path`
  - New dependency: `onnx`

Phase 5 is now complete.

## Phase 6 — Further ecosystem packages *(complete)*

- [x] Reconstruction
  - [x] `miai_reconstruction.kspace`: `simulate_kspace` / `reconstruct_from_kspace` --
    MRI reconstruction from (simulated) k-space via `torch.fft`, no new dependency for
    the core algorithm; `build_undersampling_mask` / `apply_undersampling` -- zero-filled
    reconstruction from an undersampled (accelerated-MRI-style) acquisition, center
    fraction always fully sampled
  - [x] `miai_reconstruction.metrics`: `reconstruction_quality` -- PSNR/SSIM via
    scikit-image (new dependency), for evaluating reconstruction quality (distinct from
    `miai_evaluation`'s Dice/Hausdorff, which compare segmentation masks, not images)
  - [x] `miai_reconstruction.run`: `run_kspace_reconstruction` -- SimpleITK-only I/O,
    simulates k-space from an existing volume and reconstructs it (same "simulate, then
    invert" approach `miai_diffusion` uses, since MIAI's datasets are NIfTI/DICOM, not
    raw scanner k-space)
  - [x] `miai_pipeline.stages.reconstruction.ReconstructionStage`: optional pipeline
    stage, writing `reconstructed_paths`
  - New dependency: `scikit-image`
- [x] Visualization
  - [x] `miai_visualization.slices`: `plot_slice` (single slice, optional mask overlay)
    / `plot_montage` (grid of evenly-spaced slices)
  - [x] `miai_visualization.comparison`: `plot_comparison` -- side-by-side images plus
    optional absolute-difference maps (e.g. original vs. reconstructed/denoised)
  - [x] `miai_visualization.curves`: `plot_training_curves` -- one line per metric from a
    CSV training log
  - [x] `miai_visualization.metrics`: `plot_metric_summary` -- bar or box plot of a
    per-case metric (Dice, PSNR, SSIM, ...); visualizes values already computed
    elsewhere, does not compute metrics itself
  - [x] `miai_visualization.embeddings`: `plot_embedding_projection` -- 2-component PCA
    (via `torch.linalg.svd`, no scikit-learn dependency) scatter plot of embeddings,
    e.g. from `miai_foundation_models`
  - [x] `miai_pipeline.stages.visualization.VisualizationStage`: optional pipeline stage
    writing a QC slice montage per case, `qc_visualization_paths`
  - Every plot is saved as a file (matplotlib's non-interactive "Agg" backend), never
    shown interactively -- consistent with MIAI's reproducibility-first design
  - New dependency: `matplotlib`

## Project infrastructure improvements

Not tied to a specific package -- maintenance/tooling work done after a project
state review at the end of Phase 6:

- [x] `py.typed` markers (PEP 561) added to all 13 implemented packages, so external
  consumers of MIAI get type-checking benefits from mypy, not just internal CI.
  Verified the markers actually land in a built wheel (hatchling includes them by
  default, no extra config needed).
- [x] `miai_pipeline.cli`: `miai-pipeline` console script (`run` / `validate` /
  `list-stages` subcommands) -- a pipeline YAML config can now be run from the
  terminal without writing Python, matching the "configuration over code" principle.
- [x] Fixed `examples/README.md` and `scripts/README.md`, both still worded as if the
  project were in Phase 0/1.
- [ ] PyPI packaging/publishing workflow (paused -- explicit user decision, revisit later)
- [x] `cache: pip` in `actions/setup-python` for both `ci.yml` and `security.yml`,
  keyed on `pyproject.toml` -- the install step (torch + monai are the bulk of it)
  now reuses GitHub's Actions cache on unchanged dependencies instead of
  reinstalling from PyPI on every single run.
- [x] Dependency update/security automation (Dependabot for `pip` + `github-actions`;
  weekly `pip-audit` scan against the installed environment, plus on any
  `pyproject.toml` change, in a separate `security.yml` workflow that doesn't block
  the main `lint-and-test` CI). CodeQL was not added: it requires GitHub Advanced
  Security for private repositories, which this repo doesn't have; `pip-audit`
  covers the actual risk here (vulnerable dependencies), since MIAI has no
  custom C extensions or unsafe deserialization of untrusted input to scan for.
- [x] Coverage threshold enforcement: `--cov-fail-under=95` in pytest's `addopts`
  (current coverage is ~98%, so this catches real regressions while leaving
  headroom for hard-to-cover fix-forward error paths).
- [x] Expanded `miai_evaluation.metrics`: added IoU, sensitivity, specificity
  (via MONAI's `MeanIoU`/`ConfusionMatrixMetric`), and volume similarity
  (computed directly -- MONAI has no built-in metric for it) alongside the
  existing Dice/HD95. All four are opt-in flags on `MetricsConfig`, defaulting
  to `False`, so no existing config or report format changes.
- [x] Dependency lockfile: `locks/requirements-lock.txt`, generated via
  `uv pip compile pyproject.toml --extra dev --universal --python-version 3.11`.
  `ci.yml`/`security.yml` install from it instead of resolving fresh against
  `pyproject.toml`'s lower bounds every run. Regeneration is manual (see
  `docs/coding_standards.md`). Originally lived at the repo root; moved
  under `locks/` after Dependabot auto-discovered it as a manageable
  requirements file and flooded the repo with failing PRs bumping
  individual transitive/CUDA (`nvidia-*`) pins one at a time (see
  `docs/coding_standards.md`, "Lockfile", for the full story).
- [x] `mypy tests` in CI, alongside `mypy src`. Fixed real pydicom UID-typing
  bugs in `tests/conftest.py` this surfaced (a variable reassignment that
  widened a `UID` to plain `str`, which also masked a latent bug where
  `dataset.SOPInstanceUID` could end up `None`), added missing type
  annotations to several test helpers, and added a `[[tool.mypy.overrides]]`
  block relaxing `disallow_untyped_calls` for the test modules that call
  SimpleITK directly (mirrors the existing src-side override).
- [x] Docstring-completeness linter: `interrogate` in CI (`interrogate src`),
  configured via `[tool.interrogate]` in `pyproject.toml`
  (`style = "google"`, `fail-under = 85`, current actual 87.1%). Checks
  docstring *presence* per `docs/api_design.md`'s Documentation contract.
- [x] Deepened the docstring linter beyond presence to completeness:
  `pydocstyle` added to CI, configured via `[tool.pydocstyle]` in
  `pyproject.toml` (`convention = "google"`, scoped to `src/` like
  `interrogate`). Fixed the ~40 real violations it surfaced -- mostly
  every concrete pipeline stage's `__init__`/`run` methods lacking their
  own docstring (the class-level Reads/Writes contract doesn't substitute
  for a per-method one under pydocstyle), plus malformed module/class
  docstring summaries and undocumented magic methods elsewhere.
- [x] Deepened `tests/test_pipeline_end_to_end.py`: previously only chained
  `dicom_to_nifti -> preprocessing -> dataset` (3 of ~13 stages) via a real
  `Pipeline.from_config` run. Added: the same data-prep chain extended through
  `[registration]`; a full ML-half chain (`dataset -> training -> inference ->
  evaluation`); and two optional-stage chains
  (`diffusion_training -> denoising`, `training -> export`) -- the latter two
  specifically check that a context key one stage writes
  (`model_checkpoint_path`, `diffusion_checkpoint_path`) is picked up
  automatically by the next stage, which no test had exercised before (every
  prior stage test built its context by hand rather than from a preceding
  stage's real output). `feature_extraction`, `reconstruction`, and
  `visualization` are now also chained together in
  `test_end_to_end_reconstruction_feature_extraction_visualization`
  (`dicom_to_nifti -> preprocessing -> {reconstruction, feature_extraction,
  visualization}`), confirming all three consume a real preceding
  `PreprocessingStage`'s output rather than a hand-built context.
- [x] Training tests now verify real learning, not just that training runs:
  `test_train_model_actually_learns_to_segment`
  (`tests/test_segmentation_train.py`) and
  `test_train_diffusion_model_actually_learns`
  (`tests/test_diffusion_train.py`) each compare a trained model against a
  freshly initialized model of the same architecture on the same evaluation
  data -- Dice for segmentation, noise-prediction MSE for diffusion -- so a
  training loop that silently stops updating weights (e.g. a broken optimizer
  step) would now fail CI instead of passing a "ran without crashing" check.

## Phase 7 -- miai-examples

The last package in `docs/architecture.md`'s ecosystem diagram still marked
`[planned]`. Scoped after a project-state review (2026-07-24): end-to-end
example workflows combining several MIAI packages, no new dependencies.
Approved scope, three deliverables:

- [x] `examples/configs/pipeline.yaml`: a real, runnable pipeline config for
  the full main clinical workflow (DICOM -> NIfTI -> Preprocessing -> Dataset
  -> Training -> Inference -> Evaluation). Closes a dangling reference --
  `README.md` and the `miai-pipeline` CLI usage snippet already pointed at
  `configs/pipeline.yaml` as if it existed.
- [x] `examples/segmentation_pipeline.py`: a runnable script that generates
  synthetic DICOM images and NIfTI labels (standalone -- does not import
  from `tests/conftest.py`, which is test-only) and runs the config above
  end to end via `Pipeline.from_config`. 10 synthetic cases, named
  `case_000`.. so DICOM series discovery order matches generated label
  order without inspecting UIDs.
- [x] `examples/diffusion_denoising.py`: a runnable script demonstrating an
  optional stage outside the main segmentation workflow (DDPM training +
  denoising), showing the ecosystem extends beyond the reference pipeline.
  Uses `miai_diffusion`'s package API directly rather than a pipeline YAML
  config, since this stage is meant to be used standalone.
- [x] Updated `examples/README.md` describing each example and how to run
  it, replacing the current stub.

**Phase 7 is complete.** All 14 packages in `docs/architecture.md`'s
ecosystem diagram are now implemented -- none remain marked `[planned]`.
PyPI packaging/publishing remains explicitly paused (see "Project
infrastructure improvements" above).

## Phase 8 -- `miai-segmentation` modality expansion *(complete)*

`miai-segmentation` originally offered a single reference architecture
(MONAI `UNet`, implicitly 3D). Scoped after a project-state review
(2026-08-17) to organize the package by imaging modality --
`miai_segmentation.<modality>` -- and add the architectures most
representative of each, so an experiment picks a modality and
architecture from YAML instead of the package offering only one model.

- [x] `miai_segmentation.three_d`: reorganized the existing UNet
  reference model (unchanged behavior) plus training
  (`miai_segmentation.three_d.train`) and sliding-window inference
  (`miai_segmentation.three_d.infer`, dimension-agnostic in practice)
  into their own subpackage. Added `SegResNetConfig`/`build_segresnet`
  (`monai.networks.nets.SegResNet`, Myronenko 2018) as a second 3D
  architecture, and `ArchitectureConfig`/`build_model` as a single
  dispatch point so `TrainingStage`/`InferenceStage`/`ExportStage`
  (`miai_pipeline.stages.*`) depend on one config field
  (`architecture`, replacing the old `unet` field) regardless of which
  3D architecture a run picks. **Wired into the pipeline stages.**
- [x] `miai_segmentation.two_d`: per-slice 2D architectures --
  `UNetConfig`/`build_unet` (`spatial_dims=2`) and
  `AttentionUnetConfig`/`build_attention_unet`
  (`monai.networks.nets.AttentionUnet`, Oktay et al. 2018), dispatched
  via `ArchitectureConfig`/`build_model`, same pattern as `three_d`.
  Training re-exports `miai_segmentation.three_d.train` unchanged (the
  loop is dimension-agnostic); inference is a 2D-window variant of
  `three_d.infer`. **Wired into the pipeline stages.**
- [x] `miai_segmentation.two_half_d`: 2.5D (stacked-adjacent-slice)
  architecture -- `StackedUNetConfig`/`build_stacked_unet` (a 2D UNet
  whose `in_channels` is the number of stacked adjacent slices,
  predicting the center slice's mask), dispatched via
  `ArchitectureConfig`/`build_model` for consistency even though only
  one architecture exists today. Training re-exports
  `miai_segmentation.three_d.train`; inference re-exports
  `miai_segmentation.two_d.infer` (spatially identical -- both are a 2D
  sliding window; only the model's channel count differs, which the
  model itself handles). **Wired into the pipeline stages.**
- [x] Pipeline wiring: `miai_segmentation.modality` (new, internal --
  not re-exported from `miai_segmentation`'s root `__init__.py`) adds
  `SegmentationModalityConfig`/`build_model_for_modality` and
  `SegmentationInferenceConfig`/`inference_config_for_modality`, the
  "declare all three modalities' configs, select one via a `modality`
  literal field" dispatch `TrainingStageConfig.architecture`,
  `InferenceStageConfig.architecture`/`.inference`, and
  `ExportStageConfig.architecture` (`miai_pipeline.stages.*`) now use
  instead of hardcoding `miai_segmentation.three_d`. Since 2D/2.5D
  operate per-slice rather than per-whole-volume,
  `miai_datasets.slices.expand_to_slice_dicts` (new) expands each
  case's data dict into one dict per slice (reading each volume's depth
  from its file header only, via `SimpleITK.ImageFileReader`) before
  `build_dataset` runs, and two new transforms,
  `miai_transforms.slice_transforms.ExtractSliced`/`ExtractSliceStackd`
  (registered in `TRANSFORM_REGISTRY` as `extract_slice`/
  `extract_slice_stack`), reduce a loaded `(C, D, H, W)` volume array to
  the 2D (or stacked-2D) slice a slice-level model expects.
  `miai_segmentation.two_d.infer.run_case_inference` (also re-exported
  from `two_half_d.infer`) consumes a slice-level `DataLoader` and
  reassembles each case's per-slice predictions back into one `(D, H,
  W)` volume, so `InferenceStage`'s one-prediction-file-per-case
  contract is unchanged regardless of modality -- downstream stages
  (evaluation, registration, visualization) need no changes. See
  `docs/user_guide.md` for a `modality: two_d`/`two_half_d` YAML
  example.

**Scope boundary, explicit:** `docs/compatibility_policy.md` allows
pre-1.0 breaking changes with just a MINOR version bump -- applied here
to `TrainingStageConfig.architecture`/`InferenceStageConfig.architecture`/
`InferenceStageConfig.inference`/`ExportStageConfig.architecture`'s
field types, which now wrap the previous single-modality config in a
`modality`-selecting one (existing YAML configs need
`architecture: {kind: ..., unet: ...}` rewritten as
`architecture: {modality: three_d, three_d: {kind: ..., unet: ...}}`,
and `inference: {roi_size: ..., ...}` as
`inference: {three_d: {roi_size: ..., ...}}`).

## Real-data validation

- [x] Validated the full `miai_pipeline` (preprocess -> split -> train ->
  inference -> evaluate) end to end against real clinical MRI -- the
  public ACDC (Automated Cardiac Diagnosis Challenge) cardiac cine-MRI
  dataset -- rather than only the synthetic volumes every earlier
  test/example used. Simplified in scope by design (binary "whole
  heart" segmentation rather than multi-class, one frame per patient, a
  30-patient subset -- see `docs/real_data_validation.md` for the full
  rationale), and found + fixed two real gaps synthetic data never
  exercised: independently-resampled image/label geometry drift, and
  the lack of any divisible-by-stride padding for a 3D UNet on
  arbitrarily-sized real volumes (`miai_transforms.compose
  .TRANSFORM_REGISTRY` gained a general `"divisible_pad"` entry as a
  result). Full writeup, including the honest result -- the pipeline
  is proven sound, but the toy-scale 18-patient training run overfits
  (val Dice 0.83, held-out test Dice 0.09) as expected -- in
  `docs/real_data_validation.md`. Runnable via
  `examples/validate_acdc.py`.
- [x] Second iteration: scaled up every improvement lever at once (ED+ES
  frames with a patient-level split, stronger augmentation, a deeper
  UNet, finer spacing, more epochs). Found and fixed a real
  sliding-window `roi_size` bug that had been silently undercutting
  test-time inference. Honest result even after the fix: mean test
  Dice (0.082) is statistically unchanged from the first iteration's
  0.088, despite 3.3x the data and a leak-proof split -- val Dice
  reaching 0.72 shows the bottleneck is training data volume, not
  model capacity. See `docs/real_data_validation.md`.
- [x] Third iteration: switched from a 3D UNet to MIAI's per-slice 2D
  UNet (`"two_d"` modality), matching ACDC's actual acquisition
  geometry (a stack of independent 2D short-axis slices, not a true
  volumetric scan) instead of forcing 3D convolutions across slices
  that were never spatially coherent to begin with. Same data,
  patient-level split, and epoch budget as the second iteration --
  only the modality changed. Mean test Dice jumped from ~0.08 to
  **0.71**, the first ACDC iteration where the model actually
  generalizes. See `docs/real_data_validation.md`.
- [x] Fourth iteration: scaled `DEFAULT_PATIENTS` from a 50-patient
  subset to the full 150-patient ACDC dataset (300 cases, both ED and
  ES), keeping the third iteration's 2D per-slice architecture,
  patient-level split, and augmentation unchanged (epoch budget
  reduced 40 -> 25 given ~3x the per-epoch example count). Mean test
  Dice rose from 0.71 to **0.82**, confirming more data still helps
  substantially once the model's inductive bias is correctly matched
  to the data's real acquisition geometry. See
  `docs/real_data_validation.md`.
- [x] Fifth iteration: added genuine multi-class support to
  `miai_segmentation`/`miai_evaluation` (`num_classes` on
  `TrainingConfig`/`InferenceConfig`/`MetricsConfig`, defaulting to `1`
  so every existing binary caller is unaffected), then put it to use
  training on ACDC's real 4-class ground truth (background, RV,
  myocardium, LV) instead of a merged binary "whole heart" label --
  same 150-patient dataset, architecture, split, and epoch budget as
  the fourth iteration. Macro test Dice **0.72**, with a per-class
  breakdown (RV 0.58, Myo 0.72, LV 0.86) showing the difficulty
  concentrates in the RV, not spread evenly -- a more clinically
  meaningful result than a single binary number. See
  `docs/real_data_validation.md`.
- [x] Sixth iteration: added explicit regularization to
  `miai_segmentation` (`dropout` on `UNetConfig`, `weight_decay` on
  `TrainingConfig`, both defaulting to `0.0`/off so no existing caller
  is affected), motivated by the fifth iteration's late training
  instability (validation Dice collapsed to 0.0 at epoch 23). Same
  150-patient multi-class dataset, architecture, split, and epoch
  budget as the fifth iteration, with `dropout=0.2`/`weight_decay=1e-5`.
  The instability is resolved (no collapse anywhere in this run), and
  macro test Dice improved **0.72 -> 0.75**, concentrated in the RV
  (0.58 -> 0.70). See `docs/real_data_validation.md`.
- [x] Seventh iteration: extended `miai_evaluation`'s per-class
  breakdown from Dice-only to every opted-in metric (Hausdorff
  distance, IoU, sensitivity, specificity, volume similarity), each
  computed on that class's one-hot channel the same way `dice_class_
  {c}` already was. No new training run -- re-scored the sixth
  iteration's checkpoint. Result: IoU/sensitivity mostly agree with
  Dice's RV-is-hardest story, but volume similarity disagrees -- RV is
  its worst-scoring structure while Myo is its best, the opposite
  ranking from every overlap-based metric. See
  `docs/real_data_validation.md`.
- [x] Eighth iteration: added early stopping to `miai_segmentation`
  (`TrainingConfig.early_stopping_patience`, defaulting to `None`/off),
  motivated by the sixth iteration's best checkpoint landing early
  (epoch 13 of 25) with no further improvement afterward. Raised
  `--max-epochs` to 50 with `early_stopping_patience=10`, otherwise
  identical to the sixth iteration. Training used the extra room (new
  best val Dice 0.8274 at epoch 19) before stopping at epoch 29. Result:
  macro test Dice improved **0.75 -> 0.77**, narrowing the gap to the
  binary-era ceiling (0.82) from 0.10 to 0.05, with the biggest single
  gain in Hausdorff distance (46.2mm -> 28.4mm). See
  `docs/real_data_validation.md`.
- [x] Ninth iteration: added cosine-annealed learning rate scheduling
  to `miai_segmentation` (`TrainingConfig.cosine_annealing`/
  `.min_learning_rate`, both defaulting to off/unchanged behavior),
  motivated by the eighth iteration's late-training oscillation (a dip
  to 0.7757 at epoch 28). Decayed from 1e-3 to 1e-5 over the full
  50-epoch budget, otherwise identical to the eighth iteration.
  Training ran the full budget with no early stopping and no
  oscillation, reaching a new best val Dice of 0.8576 (up from 0.8274).
  Result: macro test Dice was essentially unchanged (0.77 -> 0.77), and
  Hausdorff distance got worse across every class (macro HD95 28.4mm
  -> 35.2mm) while sensitivity improved (0.80 -> 0.84) -- a validation
  win that did not transfer to the test set. See
  `docs/real_data_validation.md`.
- [x] Tenth iteration: added `ResAttentionUNet` to `miai_segmentation.
  two_d.models` (`ArchitectureConfig.kind="res_attention_unet"`,
  defaulting to the existing `"unet"` -- no existing config affected),
  combining MONAI's `ResidualUnit` encoder/decoder with attention-gated
  skip connections built on MONAI's public primitives. Switched the
  ACDC script to this architecture and reverted the ninth iteration's
  cosine annealing to isolate architecture as the sole variable versus
  the eighth iteration. Training reached a new-fastest best val Dice
  of 0.8376 at epoch 22 (early-stopped at epoch 32). Result: macro test
  Dice fell (0.77 -> 0.76) and Hausdorff distance got substantially
  worse (28.4mm -> 37.8mm), concentrated in the right ventricle (RV
  Dice 0.70 -> 0.65, RV HD95 34.1mm -> 57.7mm) -- a validation-set win
  that generalized worse than the plain regularized UNet, the first
  iteration with a worse rather than merely flat test-set result. See
  `docs/real_data_validation.md`.
- [x] Eleventh iteration: added `ResAttentionUnetConfig.
  attention_reduction: int = 2` (matching the tenth iteration's
  previously-hardcoded bottleneck width -- no existing config
  affected) and set it to `1` in the ACDC script, motivated by the
  tenth iteration's RV-specific damage and the hypothesis that a
  narrower attention-gate bottleneck had cost RV fine-grained
  information. Training hit a late validation-Dice collapse (0.79 ->
  0.57 at epoch 21) not seen in the eighth/ninth/tenth iterations,
  early-stopping with its best checkpoint from epoch 13 (val Dice
  0.8192). Result: the hypothesis did not hold -- macro test Dice fell
  further (0.76 -> 0.73, the worst multi-class result in this
  project), RV Hausdorff distance got worse rather than better (57.7mm
  -> 60.5mm), and new damage appeared in myocardium and LV (untouched
  by the tenth iteration's problem). Three consecutive post-eighth
  changes have now each underperformed the eighth iteration's plain
  regularized UNet, still the best-performing configuration found. See
  `docs/real_data_validation.md`.
- [x] Twelfth iteration: added `ResAttentionUnetConfig.
  use_attention: bool = True` (default preserves every existing
  config/call site's behavior -- no existing config affected) and set
  it to `False` in the ACDC script, isolating whether attention itself
  or the residual-block architecture underneath it was responsible for
  the tenth/eleventh iterations' results. Validation Dice reached
  0.8278 at epoch 15, the best of any iteration so far, then training
  hit a late-training collapse (0.81 -> 0.55 at epoch 23) despite this
  run having zero attention gates -- ruling out attention as that
  instability's cause. Early-stopped at epoch 25 with its best
  checkpoint from epoch 15. Result: macro test Dice fell further still
  (0.73 -> 0.72, now the worst multi-class result in this project),
  with damage spread across myocardium and LV rather than concentrated
  in RV. Four consecutive post-eighth changes have now each
  underperformed the eighth iteration's plain regularized UNet, still
  the best-performing configuration found. See
  `docs/real_data_validation.md`.
- [x] ACDC results visualization: twelve iterations had been reported
  as text/tables only. Added `--visualize` to `examples/
  validate_acdc.py` (off by default -- wires the existing
  `VisualizationStage` into the ACDC pipeline for the first time) and
  a new `examples/visualize_acdc_results.py` script producing
  training-curve, ground-truth-vs-prediction, and per-iteration
  metric-summary plots from a completed run's outputs via
  `miai_visualization`, previously only used by the generic pipeline
  demo. See `docs/real_data_validation.md`'s "Visualizing results".
- [x] Thirteenth iteration: added `TrainingConfig.class_weights:
  tuple[float, ...] | None = None` (default preserves every existing
  config's unweighted `DiceLoss` behavior -- no existing call site
  affected), wired straight through to `DiceLoss`'s own `weight` arg
  and length-validated against the loss's channel count before
  training starts. Reverted the ACDC script to the eighth iteration's
  exact plain `UNet` baseline and set `class_weights=(0.5, 2.0, 1.5,
  1.0)` for (background, RV, myocardium, LV), motivated by RV/Myo
  being the consistently weakest structures since the seventh
  iteration. Training reached a new project-best validation Dice of
  0.8378 at epoch 44 (full 50-epoch budget, no early stopping). Result:
  macro test Dice fell (0.7740 -> 0.7348) rather than improved -- RV
  stayed essentially flat despite the highest weight, myocardium got
  slightly worse despite also being up-weighted, and LV (left at
  weight 1.0) took the largest hit of the three (0.8676 -> 0.7861). A
  useful negative result, but the eighth iteration's plain, unweighted
  UNet remains the best-performing configuration found. See
  `docs/real_data_validation.md`.
- [x] Fourteenth iteration: added `TrainingConfig.gradient_clip_norm:
  float | None = None` (default preserves every existing config's
  optimizer step byte-for-byte -- no existing call site affected),
  calling `torch.nn.utils.clip_grad_norm_` right after
  `loss.backward()` and before `optimizer.step()` when set. Reverted
  `class_weights` to `None` (the thirteenth iteration's weighting made
  things worse) and set `gradient_clip_norm=1.0` on the eighth
  iteration's plain UNet baseline, motivated by the late-training
  validation-Dice collapses seen in the eleventh and twelfth
  iterations. Training reached a near-record validation Dice of 0.8373
  at epoch 28 with no collapse anywhere in the run -- including at
  epoch 23, the exact epoch the twelfth iteration collapsed at.
  Result: macro test Dice recovered most of the thirteenth iteration's
  loss (0.7348 -> 0.7565) but stayed below the eighth iteration's
  baseline (0.7740), and Hausdorff distance got meaningfully worse
  across every structure (macro 28.4mm -> 44.1mm) -- the same
  validation-improves/boundary-suffers pattern the ninth iteration's
  cosine annealing showed. The eighth iteration's plain, unclipped,
  unweighted UNet remains the best-performing configuration found. See
  `docs/real_data_validation.md`.
- [ ] Fifteenth iteration: attempted a deeper, wider, non-residual UNet
  (`num_res_units=0`, doubled channel widths, one extra downsample
  level) on top of the eighth iteration's baseline, motivated by six
  consecutive training-procedure-only levers all underperforming it.
  No result -- the full 150-patient training run was relaunched five
  times and never survived past a few minutes; the sandbox itself
  restarted mid-run on four of five attempts, a more severe failure
  than an ordinary process crash. No checkpoint/resume support meant
  every restart lost all progress. Left unanswered, not ruled out. See
  `docs/real_data_validation.md`.
- [x] Sixteenth iteration: reverted `_ARCHITECTURE`/`_DIVISIBLE_K` to
  the eighth iteration's baseline and widened the patient-level split
  from 90/30/30 to 120/15/15 train/val/test patients (new
  `_VAL_FRACTION`/`_TEST_FRACTION` module constants, 0.1/0.1).
  Training completed cleanly (early-stopped at epoch 35/50, no sandbox
  instability this run). Result: macro test Dice fell to 0.7367 (from
  0.7740 on the eighth iteration's own 30-patient test set) with wider
  per-case variance (stdev 0.121 vs. 0.09) -- evidence that a single
  fixed-seed 15-30-patient test split is a noisy estimator on its own,
  not necessarily that more training data hurt. See `docs/
  real_data_validation.md`.
- [ ] Seventeenth iteration (planned): k-fold cross-validation for the
  ACDC validation pipeline -- train/evaluate across several
  patient-level folds and report the spread (mean +/- stdev across
  folds), not a single number from one arbitrary split. Does not exist
  yet in this codebase.
- [ ] Eighteenth iteration (planned): leave-one-patient-out (LOPO)
  evaluation, once k-fold cross-validation's infrastructure exists --
  the natural next step in reducing dependence on any one fixed
  train/val/test partition. Does not exist yet in this codebase.

## Working principle

We do not start a phase's package until the previous phase's foundations are
merged and tested. `miai-core` must be stable before `miai-dicom` depends on
it, and so on down the chain.
