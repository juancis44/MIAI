# User Guide

This is the practical, task-oriented companion to the README's quick start.
The README gets you installed and shows three isolated snippets;
this guide walks through building a real end-to-end pipeline config, then
gives a one-page reference for every package so you know where to look next.
For the *design rationale* behind these choices (why config-driven, why
typed exceptions, why one package per concern), see
[vision.md](vision.md), [architecture.md](architecture.md), and
[api_design.md](api_design.md) instead -- this guide only covers *how to
use* what's already built.

## Who this is for

You've installed MIAI (see the README's "Installation" section) and have,
or are about to have, some DICOM series and want to go from raw images to a
trained segmentation model and its evaluation metrics, without hand-writing
that orchestration yourself. If you instead want to use one MIAI package on
its own inside a larger, non-MIAI codebase (e.g. just `miai-dicom`'s
anonymization, or just `miai-registration`), skip to
["Using one package standalone"](#using-one-package-standalone).

## Core concepts

MIAI's central idea (see [vision.md](vision.md), "Philosophy") is that a
research pipeline should be **defined by a config file, not by editing
code**. Three types make this work, and understanding them makes everything
else in this guide predictable:

- **`MIAIBaseConfig`** (`miai_core`): every config object in MIAI --
  `TransformConfig`, `TrainingConfig`, `ArchitectureConfig`, ... -- is a
  Pydantic model subclassing this. It validates on construction (wrong
  type, missing required field, or an out-of-range value fails immediately
  and loudly, per `api_design.md`'s "fail loudly" principle) and loads from
  YAML via `.from_yaml(path)`.
- **`PipelineStage`** (`miai_pipeline`): one step of the clinical workflow
  (`dicom_to_nifti`, `training`, `evaluation`, ...). Each stage declares
  what it reads from and writes to a shared context -- see each stage
  class's docstring (e.g.
  `miai_pipeline.stages.training.TrainingStage`) for its exact Reads/Writes
  contract.
- **`PipelineContext`** (`miai_pipeline`): a typed key-value store passed
  from stage to stage. `DicomToNiftiStage` writes `nifti_paths`;
  `PreprocessingStage` reads them and writes `preprocessed_paths`; and so
  on down the chain. You only need to `context.set(...)` the handful of
  keys no stage produces (typically `dicom_dir` and `label_paths` -- see
  below).

## Walkthrough: the full clinical workflow

This mirrors [`examples/configs/pipeline.yaml`](../examples/configs/pipeline.yaml)
and [`examples/segmentation_pipeline.py`](../examples/segmentation_pipeline.py),
which you can run as-is (`python examples/segmentation_pipeline.py`) to see
every step below execute against a small synthetic dataset in under a
minute on CPU. This section explains *why* the config looks the way it
does, so you can adapt it to your own data.

The workflow is:

```
DICOM -> NIfTI -> Preprocessing -> [Registration] -> Dataset -> Training -> Inference -> Evaluation
```

### 1. Two inputs only you can provide

No stage in this pipeline produces segmentation labels (DICOM itself
carries no ground-truth masks), so two context keys must be set before
running:

- `dicom_dir`: a directory containing one or more DICOM series.
- `label_paths`: one ground-truth segmentation NIfTI per DICOM series,
  **in the same order `DicomToNiftiStage` discovers them in** -- see
  `examples/segmentation_pipeline.py` for how synthetic labels are
  generated and ordered to match real series discovery order.

### 2. `dicom_to_nifti` and `preprocessing`

```yaml
stages:
  - type: dicom_to_nifti
    params:
      output_dir: output/nifti

  - type: preprocessing
    params:
      output_dir: output/preprocessed
      target_spacing: [1.0, 1.0, 1.0]
      interpolation: linear
      normalization: zscore
```

`dicom_to_nifti` (`miai_dicom` + `miai_pipeline.stages.dicom_to_nifti`)
converts each discovered series to NIfTI. `preprocessing`
(`miai_transforms` + `miai_pipeline.stages.preprocessing`) resamples to a
common voxel spacing and normalizes intensities -- see
`miai_pipeline.stages.preprocessing.PreprocessingConfig`'s docstring for
every `interpolation`/`normalization` option.

### 3. `[registration]` -- optional

Insert a `registration` stage here (between `preprocessing` and `dataset`)
to align every case onto a common fixed reference image (e.g. an atlas)
before building the dataset manifest. See `miai_registration` and
`miai_pipeline.stages.registration.RegistrationStage`. Skip this stage
entirely if your data doesn't need spatial alignment across cases.

### 4. `dataset`

```yaml
  - type: dataset
    params:
      manifest_path: output/manifest.json
      context_key: preprocessed_paths
      label_context_key: label_paths
      val_fraction: 0.3
      test_fraction: 0.3
      seed: 42
```

Splits cases into train/val/test and writes a JSON manifest
(`miai_pipeline.stages.dataset.DatasetStage`). `label_context_key` makes
each manifest entry `{"image": ..., "label": ...}` instead of a bare image
path -- required for the `training`/`inference`/`evaluation` stages that
follow.

### 5. `training`

```yaml
  - type: training
    params:
      checkpoint_dir: output/checkpoints
      architecture:
        kind: unet          # or "segresnet"
        unet:
          channels: [16, 32, 64]
          strides: [2, 2]
          num_res_units: 1
      training:
        max_epochs: 5
        learning_rate: 0.001
        val_interval: 1
        device: cpu
      dataloader:
        batch_size: 1
        cache_rate: 0.0
      train_transforms:
        transforms:
          - name: load_image
            params: { keys: [image, label] }
          - name: rand_flip
            params: { keys: [image, label], prob: 0.5, spatial_axis: 0 }
          - name: ensure_type
            params: { keys: [image, label] }
      val_transforms:
        transforms:
          - name: load_image
            params: { keys: [image, label] }
          - name: ensure_type
            params: { keys: [image, label] }
```

`architecture.kind` selects the 3D segmentation architecture -- `"unet"`
(`monai.networks.nets.UNet`) or `"segresnet"` (`monai.networks.nets.
SegResNet`) -- with the matching `architecture.unet`/`architecture.
segresnet` block configuring it (see
`miai_segmentation.three_d.models.ArchitectureConfig`). This pipeline
stage only supports the 3D modality today -- `miai_segmentation.two_d`
and `.two_half_d` exist and are usable standalone (see the per-package
reference table below), but aren't selectable from this YAML yet; see
[roadmap.md](roadmap.md), Phase 8 for why. `train_transforms`/`val_transforms` are
built from `miai_transforms.TRANSFORM_REGISTRY` -- see that module for
every registered transform name and its params.

### 6. `inference`

```yaml
  - type: inference
    params:
      output_dir: output/predictions
      architecture:
        kind: unet
        unet: { channels: [16, 32, 64], strides: [2, 2], num_res_units: 1 }
      inference:
        roi_size: [32, 32, 32]
        sw_batch_size: 1
        overlap: 0.25
        threshold: 0.5
        device: cpu
      transforms:
        transforms:
          - name: load_image
            params: { keys: [image] }
          - name: ensure_type
            params: { keys: [image] }
```

`architecture` here **must match what `training` used** -- it's not
inferred from the checkpoint. Runs sliding-window inference
(`monai.inferers.sliding_window_inference`) over the manifest's `test`
split and writes one prediction NIfTI per case.

### 7. `evaluation`

```yaml
  - type: evaluation
    params:
      report_path: output/evaluation_report.json
      metrics:
        include_dice: true
        include_hausdorff: true
        include_iou: true
```

Scores each prediction against its ground-truth label
(`miai_evaluation.evaluate_predictions`) and writes a per-case + aggregate
JSON report. `MetricsConfig` also supports `include_sensitivity`,
`include_specificity`, and `include_volume_similarity` (all opt-in,
default off).

### Running it

From Python:

```python
from miai_pipeline import Pipeline, PipelineConfig, PipelineContext

config = PipelineConfig.from_yaml("configs/pipeline.yaml")
pipeline = Pipeline.from_config(config)

context = PipelineContext()
context.set("dicom_dir", "data/raw_dicom")
context.set("label_paths", ["data/labels/case0.nii.gz", "..."])
result = pipeline.run(context)
print(result.require("prediction_paths"))
```

Or from the command line, without writing any Python:

```bash
miai-pipeline validate configs/pipeline.yaml
miai-pipeline run configs/pipeline.yaml \
    --set dicom_dir=data/raw_dicom \
    --set label_paths='["data/labels/case0.nii.gz", "..."]'
miai-pipeline list-stages   # see every registered stage type and its config
```

### Optional stages outside the main workflow

Six more stages plug into a pipeline config the same way but aren't part
of the segmentation-focused chain above -- add them where they make sense
for your workflow:

| Stage `type:` | Backed by | Purpose |
|---|---|---|
| `diffusion_training` | `miai_diffusion` | Train a DDPM denoising model |
| `denoising` | `miai_diffusion` | Denoise volumes with a trained DDPM |
| `feature_extraction` | `miai_foundation_models` | Per-volume embeddings from a pretrained vision model |
| `export` | `miai_deploy` | Export a trained model to TorchScript/ONNX + a reproducibility bundle |
| `reconstruction` | `miai_reconstruction` | Simulate/reconstruct MRI k-space |
| `visualization` | `miai_visualization` | Write QC slice montages per case |

See each stage class's docstring for its exact config and Reads/Writes
contract.

## Using one package standalone

Every package works without the pipeline -- it just means you call its
functions directly and manage your own control flow instead of describing
it in YAML. Quick reference (see each package's own docstring, via
`python -c "import miai_x; help(miai_x)"` or `docs/api_design.md`, for the
full picture):

| Package | Public API entry points | What it's for |
|---|---|---|
| `miai_core` | `MIAIBaseConfig`, `get_logger`, `configure_logging`, exception hierarchy | Shared config/logging/exceptions every other package builds on |
| `miai_dicom` | `read_dicom`, `write_dicom`, `extract_metadata`, `anonymize`, `load_series` (`DicomSeries`), `validate_dataset` | DICOM I/O, metadata, de-identification |
| `miai_transforms` | `build_transforms`, `TransformConfig`, `TRANSFORM_REGISTRY` | Config-driven MONAI + SimpleITK transform pipelines |
| `miai_datasets` | `build_dataset`, `build_dataloader`, `manifest_split_to_data_dicts` | Manifest -> MONAI `Dataset`/`DataLoader` |
| `miai_segmentation.three_d` | `ArchitectureConfig`, `build_model`, `train_model`, `run_inference` | Reference 3D segmentation (UNet, SegResNet) -- wired into the pipeline stages above |
| `miai_segmentation.two_d` | `ArchitectureConfig`, `build_model`, `train_model`, `run_inference` | Per-slice 2D segmentation (UNet, AttentionUnet) -- usable standalone, not yet wired into the pipeline stages |
| `miai_segmentation.two_half_d` | `ArchitectureConfig`, `build_model`, `train_model`, `run_inference` | 2.5D stacked-adjacent-slice segmentation (a 2D UNet over stacked slices) -- usable standalone, not yet wired into the pipeline stages |
| `miai_evaluation` | `evaluate_predictions`, `compute_case_metrics`, `MetricsConfig` | Dice/Hausdorff/IoU/sensitivity/specificity/volume-similarity scoring |
| `miai_registration` | `register_images`, `apply_transform`, `read_transform`/`write_transform` | Rigid/affine/bspline registration via SimpleITK |
| `miai_reconstruction` | `simulate_kspace`, `reconstruct_from_kspace`, `build_undersampling_mask`, `reconstruction_quality` | MRI k-space simulation/reconstruction, PSNR/SSIM |
| `miai_diffusion` | `NoiseSchedule`, `build_diffusion_unet`, `train_diffusion_model`, `denoise_volume` | From-scratch DDPM for volume denoising |
| `miai_foundation_models` | `FeatureExtractor`, `extract_embeddings_for_paths` | Pretrained-model (Hugging Face) volume embeddings, "2.5D" slice-and-aggregate |
| `miai_deploy` | `export_model`, `write_bundle` | Portable TorchScript/ONNX export + reproducibility bundle |
| `miai_visualization` | `plot_slice`, `plot_montage`, `plot_comparison`, `plot_training_curves`, `plot_metric_summary`, `plot_embedding_projection` | Non-interactive plotting, every figure saved as a file |

## Troubleshooting

**"No module named 'X'" for a MIAI package.** You likely installed a subset
of dependencies. `pip install -e ".[dev]"` from the repo root installs
everything (see the README's "Installation" section); there's no way to
install one MIAI package's dependencies without the rest yet, since this is
a single monorepo (`docs/architecture.md`, "Repository strategy").

**A stage raises a `*Error` you didn't expect.** Every MIAI exception
subclasses `miai_core.exceptions.MIAIError` and is raised with a specific
message about what precondition failed (per `api_design.md`'s "fail loudly"
principle) -- the message is meant to tell you exactly what to fix (e.g. an
empty manifest split, a config field failing Pydantic validation). If the
message genuinely doesn't explain the problem, that's a bug -- see
`SECURITY.md`/`CONTRIBUTING.md` for how to report it (use a regular GitHub
issue for non-security bugs).

**Training/inference is slow or seems to hang on CPU.** All the examples
and defaults in this guide are deliberately tiny (small channel counts,
few epochs) so they run in well under a minute on CPU. Real training runs
need a GPU and realistic `max_epochs`/architecture sizing -- MIAI itself
doesn't require a GPU (see `torch.cuda.is_available()` is fine to be
`False`), but performance will be, unsurprisingly, CPU-speed.

**Config validation fails with a Pydantic error I don't understand.** Run
`miai-pipeline validate your_config.yaml` first -- it surfaces the exact
field and constraint that failed without running anything, which is
usually clearer than the traceback from a full `run`.

## Where to go next

- [examples/](../examples/) for two more complete, runnable scripts
  (the full clinical workflow, and standalone DDPM denoising).
- [docs/roadmap.md](roadmap.md) for what's implemented vs. planned.
- [CONTRIBUTING.md](../CONTRIBUTING.md) if you want to add a feature or fix
  something.
