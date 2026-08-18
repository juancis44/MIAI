# Release Process (PyPI Publishing)

This document covers two things: the one-time setup a MIAI maintainer does
on PyPI itself (which this repository's CI cannot do -- it requires
logging into pypi.org/test.pypi.org as the project owner), and the
repeatable steps to cut and publish an actual release once that setup is
done.

**`pymiai` 0.16.0 was published to PyPI on 2026-08-18** -- the first
MIAI release: https://pypi.org/project/pymiai/0.16.0/. `pip install
pymiai` now works; `README.md`'s "Installation" section reflects that.
The dry run against TestPyPI (`pymiai==0.16.0` on test.pypi.org) was
verified first, and the real PyPI release was verified afterward by
installing `pymiai==0.16.0` from pypi.org into a clean environment and
importing `miai_core` and `miai_segmentation` (including the new
`miai_segmentation.three_d.build_model`/`ArchitectureConfig`).

## Package name: `pymiai` (decided 2026-08-18)

`pyproject.toml` names the single PyPI package `pymiai` (`name =
"pymiai"`). A built wheel bundles **all 14** import packages (`miai_core`,
`miai_dicom`, `miai_pipeline`, ... `miai_visualization`) -- confirmed
locally: `python -m build` then inspecting the wheel shows every `miai_*`
top-level package inside one `pymiai-<version>-py3-none-any.whl`. This is
a side effect of the monorepo using a single `pyproject.toml` (see
`docs/architecture.md`, "Repository strategy") rather than one
`pyproject.toml` per package, so `pip install pymiai` installs the entire
ecosystem.

Naming history: the package was first going to be `miai-core` (the same
string as the `miai_core` utilities sub-package, misleadingly implying
`pip install miai-core` only installed the utilities module rather than
all 14 packages), then `miai` -- but PyPI's pending-trusted-publisher form
rejected `miai` as "too similar to an existing project" (PyPI runs a
fuzzy/confusable-name check beyond exact PEP 503 normalization, especially
aggressive for short names, and doesn't disclose which project it
collides with). `pymiai` was tried next and accepted.

## One-time setup -- DONE (2026-08-18)

Completed by the maintainer directly on PyPI/TestPyPI and in this
repository's GitHub settings (none of this is automatable from CI):

1. PyPI and TestPyPI accounts created.
2. **Pending trusted publisher** registered for `pymiai` on both
   pypi.org and test.pypi.org (PyPI project name: `pymiai`, owner:
   `juancis44`, repository name: `MIAI`, workflow name: `publish.yml`,
   environment name: `pypi` on pypi.org / `testpypi` on test.pypi.org).
3. GitHub **Settings -> Environments** has `pypi` and `testpypi` created,
   matching `.github/workflows/publish.yml`'s `environment:` blocks.
4. The repository's visibility was switched to **public**.

No PyPI API token is created or stored anywhere -- `publish.yml` uses
OIDC (`permissions: id-token: write` + `pypa/gh-action-pypi-publish`),
which is what the trusted-publisher registration above authorizes. A
pending trusted publisher does not reserve the name or create the PyPI
project until the first actual publish happens -- so `pymiai` is not yet
a real PyPI project, only a pending registration.

## Verified locally before writing this workflow

`python -m build --sdist --wheel` was run against this repository
(2026-08-18) and confirmed to: succeed with hatchling as the build
backend (already configured in `pyproject.toml`), produce a wheel
containing all 14 `miai_*` packages, and install + import cleanly
(`pip install --no-deps <wheel>`) in a separate virtual environment. The
build step itself needs no further changes.

## Cutting a release

Everything below is now unblocked -- the one-time setup above is done.

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
   https://test.pypi.org/simple/ pymiai==X.Y.Z` in a scratch venv and
   confirm it imports.
5. **Real release:** create a GitHub Release from the `vX.Y.Z` tag (GitHub
   UI: Releases -> "Draft a new release" -> choose the tag -> "Publish
   release", or `gh release create vX.Y.Z --title vX.Y.Z --notes-from-tag`).
   Publishing the Release triggers `publish.yml`'s `publish-pypi` job
   automatically.
6. Confirm on pypi.org that the new version is live, and that
   `pip install pymiai==X.Y.Z` works in a scratch environment.

## Status

- Package name: decided and registered (`pymiai`).
- Trusted-publisher setup: done, both pypi.org and test.pypi.org.
- GitHub environments (`pypi`, `testpypi`): created.
- Repository visibility: public.
- `pyproject.toml`'s `classifiers`: `"Development Status :: 4 - Beta"`
  (bumped 2026-08-18 from `2 - Pre-Alpha`) -- all 14 packages are
  implemented and tested, but the project is still pre-1.0 (breaking
  changes allowed with just a MINOR bump per
  `docs/compatibility_policy.md`) and has no real-world install base yet,
  so `"5 - Production/Stable"` isn't accurate until `1.0.0` ships.
- **First release published: `v0.16.0`, 2026-08-18** -- TestPyPI dry run
  verified first, then the real GitHub Release triggered `publish-pypi`
  (approved via the `pypi` environment's required reviewer), and
  `pymiai==0.16.0` was confirmed live on pypi.org and installable.
- Nothing left to prepare -- the next step is actually cutting a release
  (see "Cutting a release" above), whenever the maintainer decides to.
