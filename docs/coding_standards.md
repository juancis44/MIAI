# Coding Standards

## Language and versions

- Python ≥ 3.11 across all packages.

## Style

- [PEP 8](https://peps.python.org/pep-0008/) as the baseline.
- Formatting enforced by [Black](https://black.readthedocs.io/) (line length
  100).
- Linting enforced by [Ruff](https://docs.astral.sh/ruff/) (`E`, `F`, `W`,
  `I`, `N`, `UP`, `B`, `C4`, `SIM` rule sets).
- Type hints are required on all public functions, methods, and class
  attributes. `mypy --strict` runs in CI.
- Docstrings follow the
  [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  and are required on all public modules, classes, and functions.

## Testing

- [Pytest](https://docs.pytest.org/) for all tests.
- New code must include tests; CI enforces coverage does not regress.
- Unit tests should not require GPU or network access. Slow / GPU / network
  tests are marked and can be skipped in default CI runs.

## Versioning and commits

- [Semantic Versioning](https://semver.org/) for all packages.
- [Conventional Commits](https://www.conventionalcommits.org/) for commit
  messages: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`,
  `perf:`, `build:`, `ci:`.
- Every user-facing change is recorded in the package's `CHANGELOG.md`.

## Configuration over code

Experiments and pipelines are defined through configuration files, not by
editing library code. Avoid hardcoded paths, hyperparameters, or dataset
assumptions inside `src/`.

## Dependencies

Keep dependencies minimal and justified. Prefer depending on the specific
library a task needs (e.g., SimpleITK for a registration routine) over
introducing a new dependency that duplicates existing functionality.
