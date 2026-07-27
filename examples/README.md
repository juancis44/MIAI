# Examples

End-to-end usage examples of the MIAI ecosystem, combining several
packages at once. For a single package in isolation, see the "Quick
example" sections in the root [README.md](../README.md) and each
public function/class's docstring (Google-style, per
[docs/api_design.md](../docs/api_design.md)) -- these examples are for
seeing several packages work together on a real run, start to finish.

Every script here is self-contained: it generates its own small
synthetic dataset (no real patient data, no external downloads) and
runs entirely on CPU in well under a minute. None of the
data-generation helpers import from `tests/conftest.py` -- that module
is test-only fixture code, not part of MIAI's public API, so nothing
here depends on it.

## `configs/pipeline.yaml`

A real, runnable [`miai-pipeline`](../src/miai_pipeline) config for the
full main clinical workflow:

```
DICOM -> NIfTI -> Preprocessing -> Dataset -> Training -> Inference -> Evaluation
```

Run it directly from the command line once you have a DICOM directory
and matching label paths (see `segmentation_pipeline.py` below for how
to generate both), via the `miai-pipeline` console script:

```bash
miai-pipeline validate examples/configs/pipeline.yaml
miai-pipeline run examples/configs/pipeline.yaml \
    --set dicom_dir=/path/to/dicom \
    --set label_paths='["/path/to/case0_label.nii.gz", "..."]'
```

The architecture (a 3-level UNet) and epoch count (5) are deliberately
tiny so the config finishes fast on synthetic data -- swap in a larger
`unet`/`training.max_epochs` for real datasets.

## `segmentation_pipeline.py`

Generates a small synthetic dataset (10 multi-slice DICOM series with
real pixel data, plus matching NIfTI segmentation labels) and runs it
through `configs/pipeline.yaml` end to end via
`Pipeline.from_config`, printing the resulting dataset split,
checkpoint path, prediction count, and mean evaluation metrics.

```bash
python examples/segmentation_pipeline.py
```

This is the main reference example: it's the most complete
demonstration of MIAI's "configuration over code" design (see
[docs/vision.md](../docs/vision.md)) -- the entire clinical workflow
runs from one YAML file, with the Python script only responsible for
producing the input data.

## `diffusion_denoising.py`

Demonstrates one of `miai-pipeline`'s six *optional* stages --
diffusion training and denoising -- used standalone via
[`miai_diffusion`](../src/miai_diffusion)'s package API directly,
rather than through a pipeline config (optional stages are meant to be
composed into a workflow as needed, not part of one fixed reference
pipeline).

Trains a compact 3D DDPM on synthetic volumes, then simulates a "real
noisy scan" by corrupting a known-clean volume with real
forward-diffusion noise and denoises it via reverse diffusion,
reporting the mean squared error against the known-clean volume before
and after denoising (lower is better):

```bash
python examples/diffusion_denoising.py
```

## Output

All three scripts write under `examples/output/` (gitignored, safe to
delete between runs):

- `output/nifti/`, `output/preprocessed/`, `output/checkpoints/`,
  `output/predictions/`, `output/evaluation_report.json`,
  `output/manifest.json` -- from `configs/pipeline.yaml`.
- `output/synthetic_dicom/`, `output/synthetic_labels/` -- the data
  `segmentation_pipeline.py` generates before running the pipeline.
- `output/diffusion/` -- checkpoints and clean/noisy/denoised volumes
  from `diffusion_denoising.py`.
