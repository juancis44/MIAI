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

## Phase 3 — `miai-pipeline`

Clinical workflow orchestration:

```
DICOM → NIfTI → Preprocessing → Dataset → Training → Inference → Evaluation
```

- [x] Config-driven `Pipeline` / `PipelineStage` orchestration framework
- [x] `dicom_to_nifti` stage (DICOM series → NIfTI, via SimpleITK)
- [x] `preprocessing` stage (resampling, intensity normalization)
- [x] `dataset` stage (manifest generation, train/val/test split)
- [x] `training` stage (interface defined in Phase 3; concrete implementation in Phase 4)
- [x] `inference` stage (interface defined in Phase 3; concrete implementation in Phase 4)
- [x] `evaluation` stage (concrete implementation alongside `miai-evaluation`)

## Phase 4 — Integration with MONAI

- [x] `miai-transforms`: config-driven MONAI transform pipelines
- [x] `miai-datasets`: manifest -> MONAI `Dataset`/`CacheDataset`/`DataLoader`
- [x] `miai-segmentation`: reference binary 3D segmentation (MONAI `UNet`,
  Dice loss/metric training loop, sliding-window inference)
- [x] `miai_pipeline.stages.training.TrainingStage` / `.inference.InferenceStage`
  concrete implementations, wired to the three packages above
- [x] `dataset` stage: optional `label_context_key` for supervised manifests

## `miai-evaluation` — Dice / Hausdorff distance metrics

Closes out the `evaluation` stage left open in Phase 3/4:

- [x] `miai_evaluation.metrics`: Dice similarity coefficient, Hausdorff distance (HD95 by default)
- [x] `miai_evaluation.evaluate`: file-based `evaluate_predictions`, reading prediction/ground-truth
  NIfTI pairs via SimpleITK and aggregating per-case metrics into a summary report
- [x] `miai_pipeline.stages.evaluation.EvaluationStage`: concrete implementation, replacing the
  Phase 3 interface

## Phase 5 — Advanced modules *(current)*

- [ ] Registration
- [ ] Diffusion
- [ ] Foundation models
- [ ] Deployment

## Working principle

We do not start a phase's package until the previous phase's foundations are
merged and tested. `miai-core` must be stable before `miai-dicom` depends on
it, and so on down the chain.
