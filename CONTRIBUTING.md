# Contributing to MIAI

Thank you for considering a contribution. MIAI is community-driven and aims
to stay approachable for researchers who are not primarily software
engineers, while still holding a high engineering bar.

## Getting started

```bash
git clone https://github.com/juancis44/MIAI.git
cd MIAI
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Development workflow

1. Open an issue describing the change before starting significant work.
2. Create a branch: `feature/<short-description>`, `fix/<short-description>`.
3. Write tests for new behavior. All new code should be covered by `pytest`.
4. Run the local checks before opening a pull request:

   ```bash
   black .
   ruff check .
   mypy src
   pytest
   ```

5. Follow [Conventional Commits](https://www.conventionalcommits.org/) for
   commit messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
6. Open a pull request against `main`. Describe what changed and why.

## Code style

See [docs/coding_standards.md](docs/coding_standards.md) for the full style
guide: PEP8, Black formatting, Ruff linting, type hints, Google-style
docstrings, and Semantic Versioning.

## Design principles

Every contribution should respect MIAI's core principles: reproducibility
first, modular architecture, clinical orientation, and simplicity. See
[docs/vision.md](docs/vision.md) for details.

## Code of conduct

Be respectful and constructive. This project serves a community of
researchers, clinicians, and engineers working on tools that may eventually
touch patient care — precision and honesty in reporting results matter.
