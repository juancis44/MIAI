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

- [x] Registration
  - [x] `miai_registration.register`: `register_images` -- rigid/affine/bspline registration via
    `SimpleITK.ImageRegistrationMethod` (multi-resolution gradient descent)
  - [x] `miai_registration.apply`: `apply_transform` -- propagates an estimated transform to a
    paired image (e.g. a label mask, via nearest-neighbor interpolation)
  - [x] `miai_registration.transform_io`: save/load transforms
  - [x] `miai_pipeline.stages.registration.RegistrationStage`: optional pipeline stage, aligning
    every case onto a common fixed reference image (e.g. an atlas) between `preprocessing` and
    `dataset`
- [x] Diffusion
  - [x] `miai_diffusion.schedule`: `NoiseSchedule` -- forward (`q_sample`) and reverse
    (`p_sample_step`) DDPM process (Ho, Jain & Abbeel 2020), linear or cosine beta schedule
  - [x] `miai_diffusion.model`: `DiffusionUNet` -- compact 3D UNet with sinusoidal timestep
    conditioning, implemented from scratch in PyTorch (no MONAI generative extension)
  - [x] `miai_diffusion.train`: `train_diffusion_model` -- standard DDPM noise-prediction
    training loop
  - [x] `miai_diffusion.denoise`: `denoise_volume` / `run_denoising` -- SDEdit-style denoising
    of real noisy volumes via partial reverse diffusion from a configurable start timestep
  - [x] `miai_pipeline.stages.diffusion_training.DiffusionTrainingStage` /
    `.denoising.DenoisingStage`: optional pipeline stages, outside the main segmentation
    workflow
- [x] Foundation models
  - [x] `miai_foundation_models.extractor`: `FeatureExtractor` -- embeds a volume with a
    pretrained Hugging Face Hub vision model (default `facebook/dinov2-small`) using a
    slice-and-aggregate ("2.5D") strategy: run the 2D model over every slice, pool tokens
    per slice (CLS or mean), then pool across slices (mean or max) into one embedding
  - [x] `miai_foundation_models.run`: `extract_embeddings_for_paths` -- extracts and saves
    embeddings for a list of volumes on disk, SimpleITK-only I/O
  - [x] `miai_pipeline.stages.feature_extraction.FeatureExtractionStage`: optional pipeline
    stage, outside the main segmentation workflow, for embedding-based downstream tasks
    (retrieval, clustering, simple classification) without fine-tuning
- [x] Deployment
  - [x] `miai_deploy.export`: `export_model` -- exports a trained model to a portable
    inference format, TorchScript (`torch.jit.trace`, no extra dependency) or ONNX
    (`torch.onnx.export`); reference task is portable export, not live model serving
  - [x] `miai_deploy.bundle`: `BundleMetadata` + `write_bundle` -- packages an exported
    model together with reproducibility metadata (name/version/description) as a single
    bundle directory
  - [x] `miai_pipeline.stages.export.ExportStage`: optional pipeline stage exporting the
    trained segmentation model produced by `TrainingStage`, writing `deploy_bundle_path`
  - New dependency: `onnx`

Phase 5 is now complete.

## Working principle

We do not start a phase's package until the previous phase's foundations are
merged and tested. `miai-core` must be stable before `miai-dicom` depends on
it, and so on down the chain.
