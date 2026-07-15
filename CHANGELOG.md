# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
