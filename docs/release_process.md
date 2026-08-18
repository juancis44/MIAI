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

## Package name: `miai` (decided 2026-08-18)

`pyproject.toml` names the single PyPI package `miai` (`name = "miai"`).
A built wheel bundles **all 14** import packages (`miai_core`,
`miai_dicom`, `miai_pipeline`, ... `miai_visualization`) -- confirmed
locally: `python -m build` then inspecting the wheel shows every `miai_*`
top-level package inside one `miai-<version>-py3-none-any.whl`. This is a
side effect of the monorepo using a single `pyproject.toml` (see
`docs/architecture.md`, "Repository strategy") rather than one
`pyproject.toml` per package, so `pip install miai` installs the entire
ecosystem -- which is exactly what the name implies.

(The package was previously going to be named `miai-core` -- the same
string as the `miai_core` utilities sub-package -- which would have
misleadingly implied `pip install miai-core` only installed the utilities
module rather than all 14 packages. `miai` was chosen instead to avoid
that collision. Both names were confirmed available on PyPI as of
2026-08-18; `miai` has since been registered as the trusted-publisher
target, see below.)

## One-time setup (PyPI project owner only, not automatable from CI)

Do this once, before the first release, on both **test.pypi.org** (for dry
runs) and **pypi.org** (for real releases) -- they're separate accounts/
projects with separate trusted-publisher configuration:

1. Create an account on [pypi.org](https://pypi.org) and
   [test.pypi.org](https://test.pypi.org) if you don't have one (they're
   independent accounts).
2. Register a **pending trusted publisher** for `miai` -- PyPI supports
   this *before* the project exists, so the very first publish doesn't
   need an API token at all:
   - pypi.org: account **Publishing** settings -> "Add a new pending
     publisher" -> PyPI project name: `miai` -> owner: `juancis44` ->
     repository name: `MIAI` -> workflow name: `publish.yml` ->
     environment name: `pypi`.
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
   https://test.pypi.org/simple/ miai==X.Y.Z` in a scratch venv and
   confirm it imports.
5. **Real release:** create a GitHub Release from the `vX.Y.Z` tag (GitHub
   UI: Releases -> "Draft a new release" -> choose the tag -> "Publish
   release", or `gh release create vX.Y.Z --title vX.Y.Z --notes-from-tag`).
   Publishing the Release triggers `publish.yml`'s `publish-pypi` job
   automatically.
6. Confirm on pypi.org that the new version is live, and that
   `pip install miai==X.Y.Z` works in a scratch environment.

## What still needs a decision from the maintainer (not done by this doc)

- Actually performing the one-time PyPI/TestPyPI trusted-publisher setup
  (requires PyPI account access this repository's CI/tooling doesn't
  have) -- see "One-time setup" above.

`pyproject.toml`'s `classifiers` was bumped to `"Development Status ::
4 - Beta"` (2026-08-18): all 14 packages are implemented and tested, but
the project is still pre-1.0 (breaking changes allowed with just a MINOR
bump per `docs/compatibility_policy.md`) and has no real-world install
base yet, so `"5 - Production/Stable"` isn't accurate until `1.0.0`
ships.
