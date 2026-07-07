# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
