# Release Process (PyPI Publishing)

This document covers two things: the one-time setup a MIAI maintainer does
on PyPI itself (which this repository's CI cannot do -- it requires
logging into pypi.org/test.pypi.org as the project owner), and the
repeatable steps to cut and publish an actual release once that setup is
done.

Publishing has been prepared (this document, `.github/workflows/publish.yml`,
and a verified local `python -m build` -- see below) but **not yet
performed**: no MIAI release has been published to PyPI as of this
writing. `README.md`'s "Installation" section reflects that; update it
once the first release actually ships.

## An open decision before the first publish: the package name

`pyproject.toml` currently names the single PyPI package `miai-core`
(`name = "miai-core"`), but a built wheel bundles **all 14** import
packages (`miai_core`, `miai_dicom`, `miai_pipeline`, ...
`miai_visualization`) -- confirmed locally: `python -m build` then
inspecting the wheel shows every `miai_*` top-level package inside one
`miai_core-<version>-py3-none-any.whl`. This is a side effect of the
monorepo using a single `pyproject.toml` (see `docs/architecture.md`,
"Repository strategy") rather than one `pyproject.toml` per package.

That means `pip install miai-core` today would install the *entire*
ecosystem under a name that suggests it's just the core utilities package.
Both `miai-core` and `miai` are currently unregistered on PyPI (checked
2026-08-18), so there's room to pick either:

- **Keep `miai-core`** -- no `pyproject.toml` change needed, but the name
  stays misleading for as long as the monorepo ships as one package.
- **Rename to `miai`** -- more accurate for what actually gets installed,
  but changes the `pip install` command in every doc/example, and once
  published a PyPI name can't be renamed later (only yanked and
  re-published under a new name, which fragments the version history).

This should be decided **before** the first publish, since it's very hard
to undo after. Whichever name is chosen, register it as a trusted
publisher (below) under that exact name.

## One-time setup (PyPI project owner only, not automatable from CI)

Do this once, before the first release, on both **test.pypi.org** (for dry
runs) and **pypi.org** (for real releases) -- they're separate accounts/
projects with separate trusted-publisher configuration:

1. Create an account on [pypi.org](https://pypi.org) and
   [test.pypi.org](https://test.pypi.org) if you don't have one (they're
   independent accounts).
2. Register a **pending trusted publisher** for the chosen package name
   (see above) -- PyPI supports this *before* the project exists, so the
   very first publish doesn't need an API token at all:
   - pypi.org: account **Publishing** settings -> "Add a new pending
     publisher" -> PyPI project name: `miai-core` (or `miai`) -> owner:
     `juancis44` -> repository name: `MIAI` -> workflow name:
     `publish.yml` -> environment name: `pypi`.
   - test.pypi.org: same steps, environment name: `testpypi` instead.
3. In this GitHub repository's **Settings -> Environments**, create two
   environments named exactly `pypi` and `testpypi` (matching step 2 and
   `.github/workflows/publish.yml`'s `environment:` blocks). Optionally add
   a required reviewer to the `pypi` environment so a real publish needs a
   manual approval click, even though the workflow itself only runs on a
   published GitHub Release or explicit manual dispatch.

No PyPI API token is created or stored anywhere -- `publish.yml` uses
OIDC (`permissions: id-token: write` + `pypa/gh-action-pypi-publish`),
which is what the trusted-publisher registration above authorizes.

## Verified locally before writing this workflow

`python -m build --sdist --wheel` was run against this repository
(2026-08-18) and confirmed to: succeed with hatchling as the build
backend (already configured in `pyproject.toml`), produce a wheel
containing all 14 `miai_*` packages, and install + import cleanly
(`pip install --no-deps <wheel>`) in a separate virtual environment. The
build step itself needs no further changes.

## Cutting a release

1. Decide the version bump per `docs/compatibility_policy.md` and
   `docs/coding_standards.md` (SemVer). Update `pyproject.toml`'s
   `version` field.
2. Move `CHANGELOG.md`'s `## [Unreleased]` section to a new dated
   `## [X.Y.Z] - YYYY-MM-DD` heading, leaving a fresh empty `##
   [Unreleased]` above it -- the pattern already used for every release so
   far (e.g. commit `9a8ba08`, "cut CHANGELOG's Unreleased section as
   v0.15.0").
3. Commit (`chore: cut CHANGELOG's Unreleased section as vX.Y.Z`), push to
   `main`, and tag: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin
   vX.Y.Z`.
4. **Dry run first (recommended, at least for the first release ever):**
   trigger `.github/workflows/publish.yml` manually (Actions ->
   "Publish to PyPI" -> "Run workflow" -> target: `testpypi`), then verify
   the published version on test.pypi.org: `pip install --index-url
   https://test.pypi.org/simple/ miai-core==X.Y.Z` in a scratch venv and
   confirm it imports.
5. **Real release:** create a GitHub Release from the `vX.Y.Z` tag (GitHub
   UI: Releases -> "Draft a new release" -> choose the tag -> "Publish
   release", or `gh release create vX.Y.Z --title vX.Y.Z --notes-from-tag`).
   Publishing the Release triggers `publish.yml`'s `publish-pypi` job
   automatically.
6. Confirm on pypi.org that the new version is live, and that
   `pip install miai-core==X.Y.Z` works in a scratch environment.

## What still needs a decision from the maintainer (not done by this doc)

- The package-name decision above.
- Actually performing the one-time PyPI/TestPyPI trusted-publisher setup
  (requires PyPI account access this repository's CI/tooling doesn't
  have).
- Whether `pyproject.toml`'s `classifiers` should move off
  `"Development Status :: 2 - Pre-Alpha"` before or at the first publish
  (commonly bumped to `"4 - Beta"` for a pre-1.0 first release, `"5 -
  Production/Stable"` once 1.0 ships).
