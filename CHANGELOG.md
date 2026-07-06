# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
