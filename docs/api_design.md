# API Design

## Principles

1. **Consistent APIs across packages.** A user who has learned
   `miai-dicom`'s conventions should find `miai-segmentation` familiar.
2. **Typed, explicit interfaces.** Public functions declare their input and
   output types; avoid `*args, **kwargs` passthroughs on public APIs.
3. **Configuration objects, not long parameter lists.** Complex behavior is
   configured through a typed config object (Pydantic model) rather than
   many individual keyword arguments.
4. **Fail loudly and specifically.** Use MIAI's exception hierarchy
   (`miai_core.exceptions`) rather than bare `Exception` or silent fallbacks,
   especially around clinical data assumptions (e.g., unexpected DICOM
   orientation, missing modality).
5. **Composability over inheritance.** Prefer small, composable functions
   and dataclasses/Pydantic models over deep class hierarchies, so pipeline
   stages can be swapped independently (per the modular architecture
   principle in [vision.md](vision.md)).
6. **No hidden I/O.** Functions that read or write files, network resources,
   or DICOM/NIfTI data say so in their name and signature; pure
   transformation functions never touch disk.

## Package public surface

A package's public API is the union of its top-level `__init__.py` plus any
**documented sub-namespace** `__init__.py`s. A module not re-exported at
either level is internal and may change without a deprecation cycle
(see [compatibility_policy.md](compatibility_policy.md)); anything re-exported
at either level follows semantic versioning.

Most packages need only the top-level form: everything public lives in one
flat `__init__.py`, and that is still the default -- reach for a
sub-namespace only when a package has grown a genuine second axis of
variation (see below).

**Sub-namespaces exist for one reason: avoiding name collisions between
interchangeable variants of the same concept**, not for organizing code by
size or by feature. A package earns a documented sub-namespace when it
contains multiple mutually-exclusive implementations of the same role --
e.g. "the segmentation architecture for this experiment" (`three_d`,
`two_d`, `two_half_d`) or "the pipeline stage for this workflow step"
(`training`, `inference`, ... under `miai_pipeline.stages`) -- where a flat
namespace would force every variant's config/builder to share one name
(`UNetConfig`, `build_model`, ...) or be renamed apart with an awkward
suffix (`UNetConfig3D`, `UNetConfig2D`, ...). The sub-namespace's own name
*is* the disambiguator, so call sites stay self-documenting:
`miai_segmentation.three_d.build_model` vs. `miai_segmentation.two_d.
build_model` reads unambiguously; `miai_segmentation.build_model_3d` does
not scale past two or three variants and hides the axis of variation from
anyone scanning imports.

In that case, the top-level `__init__.py` re-exports only what's genuinely
shared across every variant (shared exceptions, orchestration types, a
registry mapping names to variants) -- never the variant-specific
configs/builders/classes themselves, which live one level down instead.
Two examples already follow this shape:

- `miai_pipeline` (root): exposes `Pipeline`, `PipelineConfig`,
  `PipelineContext`, `PipelineStage`, and the pipeline-level exceptions --
  the orchestration types shared by every stage. The concrete stage classes
  (`TrainingStage`, `InferenceStage`, `ExportStage`, ... 13 in total) are
  **not** re-exported here; they live in `miai_pipeline.stages`, which also
  builds `STAGE_REGISTRY` (mapping a config file's `type:` string to its
  stage class).
- `miai_segmentation` (root): exposes only `SegmentationError`, the one
  exception shared across every modality. Modality-specific architectures
  live under `miai_segmentation.<modality>` -- currently
  `miai_segmentation.three_d` (`UNetConfig`/`build_unet`,
  `SegResNetConfig`/`build_segresnet`, `ArchitectureConfig`/`build_model`,
  plus training/inference); `two_d` and `two_half_d` are planned
  (`docs/roadmap.md`, Phase 8) and will follow the same shape.

A sub-namespace is documented (and therefore part of the public API surface)
when its own `__init__.py` has a module docstring describing its role and
an explicit `__all__`, mirroring what a top-level package `__init__.py`
does. Don't introduce a sub-namespace preemptively for a package that only
has one implementation of a role today -- add it when (and because) a
second, mutually-exclusive implementation actually shows up, the same way
`miai_segmentation.three_d` was only split out once `SegResNet` joined
`UNet` as a second 3D architecture.

## Documentation contract

Every public function/class docstring (Google style) documents:

- Purpose (one line)
- Args, with types
- Returns, with type
- Raises, listing the specific MIAI exceptions it may raise

This is enforced by review, and docstring *presence* (not full Args/Returns/
Raises completeness) is additionally enforced in CI via `interrogate`
(`[tool.interrogate]` in `pyproject.toml`, `style = "google"`,
`fail-under = 85`).
