# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
