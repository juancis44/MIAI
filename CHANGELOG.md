# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
