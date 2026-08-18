# Architecture

## Ecosystem layout

```
MIAI
├── miai-core               # common utilities, config, logging, IO, exceptions, typing        [done]
├── miai-dicom              # DICOM read/write, metadata, anonymization, series loading        [done]
├── miai-pipeline           # clinical workflow orchestration (DICOM -> NIfTI -> ... -> eval)  [done]
├── miai-transforms         # preprocessing / augmentation transforms                          [done]
├── miai-datasets           # dataset management and wrappers                                  [done]
├── miai-segmentation       # segmentation models and training recipes                         [done]
├── miai-registration       # image registration                                               [done]
├── miai-reconstruction     # image reconstruction                                             [done]
├── miai-diffusion          # generative / diffusion models                                    [done]
├── miai-foundation-models  # pretrained-model embeddings / feature extraction                 [done]
├── miai-evaluation         # metrics and evaluation harnesses                                 [done]
├── miai-visualization      # visualization tooling                                            [done]
├── miai-deploy             # clinical deployment                                              [done]
└── miai-examples           # end-to-end example workflows                                     [done]
```

Each package is designed to be usable independently and to depend only on
`miai-core` and, where relevant, on `miai-dicom` / `miai-transforms`. No
package should require the entire ecosystem to function.

`miai-segmentation` is organized internally by imaging modality --
`miai_segmentation.three_d` (implemented), `.two_d` and `.two_half_d`
(planned) -- each exposing its own architecture configs and a
`build_model` dispatcher (see `miai_segmentation.three_d.models`), so
`ArchitectureConfig`/`build_model`-style YAML selection stays uniform
across modalities as more are added. See `docs/roadmap.md`'s Phase 8 for
progress.

## Repository strategy

MIAI is developed as a single monorepo (this repository). This was
originally meant to hold only through Phase 0-1 while the core
abstractions stabilized, but as of Phase 6 it still holds for all
thirteen packages -- coordinating shared conventions (config style,
exception hierarchy, mypy overrides) has stayed easier within one
repository than the overhead of splitting would justify so far. Each
package still lives under its own `src/<package>` namespace with its
own tests, so it remains possible to extract any of them into a
standalone repository later without restructuring its internals; no
package has been extracted yet.

## Folder philosophy

Every package, once split into its own repository, is meant to follow the
same internal layout:

```
package/
├── docs/
├── examples/
├── tests/
├── src/
├── scripts/
├── README.md
├── LICENSE
├── pyproject.toml
└── CHANGELOG.md
```

This is a target for that future split, not the current layout: while all
packages live in this one monorepo (see "Repository strategy" above), these
folders are shared at the repository root -- one `docs/`, one `tests/`, one
root `pyproject.toml`/`CHANGELOG.md` covering every package's version
together -- rather than duplicated per package under `src/<package>/`.

## Clinical workflow (implemented as of Phase 4)

```
DICOM
  ↓
NIfTI
  ↓
Preprocessing
  ↓
[Registration]   (optional)
  ↓
Dataset
  ↓
Training
  ↓
Inference
  ↓
Evaluation
```

`miai-pipeline` is responsible for orchestrating this flow, delegating each
stage to the relevant package (`miai-dicom` for the first step,
`miai-transforms` for preprocessing, `miai-registration` for the optional
alignment step, `miai-datasets` for dataset assembly, and so on), so no
single package needs to know about the others' internals.

Phases 5-6 added six more pipeline stages that sit outside this main
segmentation workflow rather than extending it: diffusion training/denoising
(`miai-diffusion`), feature extraction (`miai-foundation-models`), model
export (`miai-deploy`), k-space reconstruction (`miai-reconstruction`), and
QC visualization (`miai-visualization`). Each is optional and only runs if
included in a pipeline's YAML config -- see docs/roadmap.md, Phase 5-6, for
what each one does.

## Configuration-driven design

Reproducibility is a first-class concern: pipelines, models, and experiments
are defined by configuration files (YAML), not by editing Python code.
`miai-core` owns the configuration system (`MIAIBaseConfig`, built in Phase 1
-- see `miai_core.config`) that every other package's config classes
subclass.

## Integration with existing libraries

MIAI does not reimplement functionality already provided by an existing,
well-maintained library. As of Phase 6, the actual dependency set (see the
root `pyproject.toml`) is: PyTorch and MONAI (models, training, transforms),
SimpleITK (all image I/O -- reading, writing, resampling), NumPy and SciPy,
Pydantic and PyYAML (config), PyDICOM (DICOM read/write), Hugging Face
`transformers`/`huggingface_hub` (pretrained foundation models) with
Pillow (image preprocessing for those models), ONNX (portable model
export), scikit-image (PSNR/SSIM reconstruction-quality metrics), and
Matplotlib (plotting). MIAI packages wrap and orchestrate
these libraries behind consistent, typed, documented APIs (see
[api_design.md](api_design.md)).

Two deliberate exclusions, both decided during Phase 4 after an early
implementation accidentally pulled one in as an undeclared transitive
dependency: **NiBabel** and generic **ITK** are not used anywhere in MIAI.
All image I/O goes through SimpleITK directly instead (a project-wide
preference, not a limitation of the other libraries) -- see
`miai_transforms.sitk_transforms.LoadImageSitkd` for the custom loader this
motivated in place of MONAI's own `LoadImaged`. OpenCV is not used either;
no package has needed 2D computer-vision operations SimpleITK/scikit-image/
Matplotlib don't already cover.
