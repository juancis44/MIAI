# Architecture

## Ecosystem layout

```
MIAI
├── miai-core            # common utilities, config, logging, IO, exceptions, typing
├── miai-dicom            # DICOM read/write, metadata, anonymization, series loading
├── miai-pipeline          # clinical workflow orchestration (DICOM → NIfTI → ... → eval)
├── miai-transforms        # preprocessing / augmentation transforms
├── miai-datasets          # dataset management and wrappers
├── miai-segmentation      # segmentation models and training recipes
├── miai-registration      # image registration
├── miai-reconstruction    # image reconstruction
├── miai-diffusion         # generative / diffusion models
├── miai-evaluation        # metrics and evaluation harnesses
├── miai-visualization     # visualization tooling
├── miai-deploy            # clinical deployment
└── miai-examples          # end-to-end example workflows
```

Each package is designed to be usable independently and to depend only on
`miai-core` and, where relevant, on `miai-dicom` / `miai-transforms`. No
package should require the entire ecosystem to function.

## Repository strategy

During Phase 0 and Phase 1, MIAI is developed as a single monorepo (this
repository) so that the core abstractions can stabilize quickly without the
overhead of coordinating changes across many repositories. Each package still
lives under its own `src/<package>` namespace with its own tests, so it can
be extracted into a standalone repository later (starting with `miai-dicom`
in Phase 2) without restructuring its internals.

## Folder philosophy

Every package (whether a subfolder of the monorepo or, later, its own repo)
follows the same internal layout:

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

## Clinical workflow (target, Phase 3+)

```
DICOM
  ↓
NIfTI
  ↓
Preprocessing
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
`miai-transforms` for preprocessing, `miai-datasets` for dataset assembly,
and so on), so no single package needs to know about the others' internals.

## Configuration-driven design

Reproducibility is a first-class concern: pipelines, models, and experiments
are defined by configuration files (YAML), not by editing Python code.
`miai-core` will own the configuration system (Phase 1) that every other
package builds on.

## Integration with existing libraries

MIAI does not reimplement functionality already provided by MONAI, PyTorch,
SimpleITK, NiBabel, PyDICOM, ITK, NumPy, SciPy, OpenCV, or scikit-image.
Instead, MIAI packages wrap and orchestrate these libraries behind
consistent, typed, documented APIs (see [api_design.md](api_design.md)).
