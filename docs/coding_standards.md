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

### Lockfile

`pyproject.toml` only sets lower bounds (`torch>=2.2`, etc.) -- it defines
what MIAI is compatible with, not what CI actually installs.
`requirements-lock.txt` pins every dependency, direct and transitive, to an
exact version, and is what `ci.yml`/`security.yml` install from
(`pip install -r requirements-lock.txt`, then `pip install -e . --no-deps`
for the package itself), so every run installs identical versions instead
of whatever the resolver would pick that day.

Regenerate it after changing `pyproject.toml`'s dependencies (adding,
removing, or re-bounding a package):

```bash
uv pip compile pyproject.toml --extra dev --universal --python-version 3.11 -o requirements-lock.txt
```

`--universal` resolves for every supported OS/architecture (embedding
environment markers where versions differ, e.g. `scipy` between Python 3.11
and 3.12) rather than just the resolving machine's own platform.
Dependabot still opens PRs against `pyproject.toml`'s bounds as before; the
lockfile is not currently auto-updated by Dependabot and needs a manual
regeneration + PR when a bound changes.
