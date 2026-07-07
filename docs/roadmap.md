# Roadmap

MIAI is built step by step, one phase at a time. Each phase should be
functional and tested before moving to the next.

## Phase 0 — Project Design

- [x] Vision and mission
- [x] Ecosystem architecture
- [x] Documentation structure
- [x] Coding standards
- [x] API design principles
- [x] Repository scaffold (this commit)

## Phase 1 — `miai-core`

Common utilities every other package depends on:

- [x] Configuration system (YAML-based, Pydantic-validated)
- [x] Logging
- [x] IO utilities
- [x] Exceptions hierarchy
- [x] Typing utilities
- [x] General utilities
- [x] Test suite and CI

## Phase 2 — `miai-dicom`

- [x] Reading DICOM
- [x] Writing DICOM
- [x] Metadata extraction
- [x] Anonymization
- [x] Series loading
- [x] Validation

## Phase 3 — `miai-pipeline` *(current)*

Clinical workflow orchestration:

```
DICOM → NIfTI → Preprocessing → Dataset → Training → Inference → Evaluation
```

- [x] Config-driven `Pipeline` / `PipelineStage` orchestration framework
- [x] `dicom_to_nifti` stage (DICOM series → NIfTI, via SimpleITK)
- [x] `preprocessing` stage (resampling, intensity normalization)
- [x] `dataset` stage (manifest generation, train/val/test split)
- [ ] `training` stage (interface defined; concrete implementation in Phase 4)
- [ ] `inference` stage (interface defined; concrete implementation in Phase 4)
- [ ] `evaluation` stage (interface defined; concrete implementation alongside miai-evaluation)

## Phase 4 — Integration with MONAI

- [ ] Dataset wrappers
- [ ] Transforms
- [ ] Training utilities
- [ ] Inference

## Phase 5 — Advanced modules

- [ ] Registration
- [ ] Diffusion
- [ ] Foundation models
- [ ] Deployment

## Working principle

We do not start a phase's package until the previous phase's foundations are
merged and tested. `miai-core` must be stable before `miai-dicom` depends on
it, and so on down the chain.
