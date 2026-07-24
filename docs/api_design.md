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

Each package exposes its public API from its top-level `__init__.py` only.
Internal modules (anything not re-exported there) may change without a
deprecation cycle; the re-exported surface follows semantic versioning.

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
