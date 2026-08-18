# Compatibility and Deprecation Policy

This document defines what MIAI promises to callers about API stability,
and how a breaking change is introduced when one is genuinely needed.
`docs/coding_standards.md`'s "Versioning and commits" section says MIAI
follows [Semantic Versioning](https://semver.org/); this document is the
detail behind that promise.

## What "public API" means here

Only the surface described in `docs/api_design.md`'s "Package public
surface" section is covered by this policy: a package's top-level
`__init__.py`, plus any documented sub-namespace `__init__.py` (e.g.
`miai_segmentation.three_d`, `miai_pipeline.stages`). Anything not
re-exported at one of those two levels -- a module's internal helper
functions, private classes, implementation details reachable only by
importing the module directly (e.g. `miai_segmentation.three_d.models.
_some_helper`) -- is **not** covered and may change in any release,
including a patch release, without notice.

## Before 1.0 (current)

MIAI is `0.x.y`. Per SemVer, **any** `0.x` release may contain breaking
changes to the public API; only the `MINOR` (`x`) number needs to bump, not
`MAJOR`. Every release so far has in fact been a `MINOR` bump against a
single `main` branch, with no LTS or patch branches -- see `SECURITY.md`'s
supported-versions table. Breaking changes made during this phase (e.g. the
`unet:` -> `architecture:` config field rename in `TrainingStageConfig`/
`InferenceStageConfig`/`ExportStageConfig`) are recorded in `CHANGELOG.md`
under the release that introduced them, but do not require a deprecation
cycle -- `0.x` is explicitly the phase for shaking out the API shape before
committing to it.

## From 1.0 onward

Once a `1.0.0` release ships, the public API (as scoped above) follows
standard SemVer:

- **PATCH** (`1.0.x`): bug fixes only. No public API changes, additions, or
  removals.
- **MINOR** (`1.x.0`): backward-compatible additions only -- new
  functions/classes/config fields, new optional parameters with defaults,
  new modality sub-namespaces (e.g. a future `miai_segmentation.two_d`).
  Existing public call sites keep working unmodified.
- **MAJOR** (`x.0.0`): the only release type allowed to remove or change
  the meaning of existing public API. Requires the deprecation cycle below
  first, except for the exemptions listed under "What doesn't require a
  deprecation cycle."

### The deprecation cycle

When a `MAJOR`-worthy change is needed post-1.0 (renaming a config field,
removing a function, changing a parameter's required/optional status,
changing a return type):

1. **Introduce the replacement alongside the old API** in a `MINOR`
   release. Both work. The old path emits a `DeprecationWarning` (Python's
   `warnings.warn(..., DeprecationWarning)`) at call time, naming the
   replacement and, if there's a released `MAJOR` version it will be
   removed in, that version.
2. **Document the deprecation** in that release's `CHANGELOG.md` entry
   under a `### Deprecated` heading, and in the deprecated symbol's own
   docstring (a `.. deprecated::` note or equivalent in prose, per the
   Google docstring style already in use).
3. **Keep both paths working for at least one full `MINOR` release cycle**
   before the old path is removed in the next `MAJOR`. There is no fixed
   calendar window (MIAI does not commit to a release cadence) -- the
   requirement is *at least one intervening `MINOR` release* between
   introducing the warning and removing the old path, so anyone pinning to
   a specific `MINOR` version sees the warning before upgrading into the
   `MAJOR` that removes it.
4. **Remove the deprecated path in the next `MAJOR` release**, with a
   `### Removed` `CHANGELOG.md` entry cross-referencing the `MINOR` release
   that introduced the deprecation.

### What doesn't require a deprecation cycle

- Changes to anything outside the public API surface as scoped above
  (internal modules/helpers).
- Fixing a bug where the previous behavior was already documented as
  incorrect, or violated the function's own docstring contract (a bug fix,
  not a behavior change from the caller's documented point of view).
- Security fixes, where keeping the vulnerable behavior available under a
  deprecation window would defeat the fix's purpose. These still go through
  `SECURITY.md`'s reporting process and get a clear `CHANGELOG.md` entry,
  but may ship as a `PATCH` even if technically breaking.
- New required fields on a config object that has never been released
  publicly yet (i.e. was only ever present in an `Unreleased` `CHANGELOG.md`
  section) -- it was never part of a released public API to begin with.

## Reviewing a change against this policy

When reviewing a pull request (see `CONTRIBUTING.md`), check: does this
touch a top-level or documented-sub-namespace `__init__.py`'s re-exported
names, or change the accepted types/required-ness of an existing public
function's parameters? If yes, and MIAI has shipped `1.0` by then, it needs
either a `MINOR`-release deprecation cycle (see above) or a `MAJOR` bump
with the removal called out explicitly in `CHANGELOG.md` -- not a silent
`PATCH`/`MINOR` change.
