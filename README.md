# MIAI — Medical Image Artificial Intelligence

**An open-source ecosystem for reproducible medical imaging workflows.**

MIAI simplifies the development of reproducible, modular, and clinically-oriented
medical imaging workflows. Rather than replacing existing libraries such as
[MONAI](https://monai.io/), [PyTorch](https://pytorch.org/), or
[SimpleITK](https://simpleitk.org/), MIAI integrates them into a coherent
software architecture focused on research reproducibility, software
engineering best practices, and clinical applicability.

> **Status:** all 14 planned packages implemented --
> `miai-core`, `miai-dicom`, `miai-pipeline`, `miai-transforms`,
> `miai-datasets`, `miai-segmentation`, `miai-evaluation`,
> `miai-registration`, `miai-diffusion`, `miai-foundation-models`,
> `miai-deploy`, `miai-reconstruction`, `miai-visualization`, and
> `miai-examples`. The full clinical workflow (DICOM -> NIfTI ->
> Preprocessing -> [Registration] -> Dataset -> Training -> Inference
> -> Evaluation) runs end to end, plus optional DDPM-based volume
> denoising, pretrained-model embedding extraction, portable model
> export (TorchScript/ONNX), MRI k-space reconstruction, and plotting
> tools for volumes/comparisons/curves/metrics/embeddings. See
> [examples/](examples/) for runnable end-to-end scripts and
> [docs/roadmap.md](docs/roadmap.md) for what's next.

> **Disclaimer:** MIAI is research and engineering tooling, not a medical
> device. It has not been evaluated or cleared by any regulatory body (FDA,
> CE marking, or equivalent), and no model, pipeline, or output produced
> with it should be used for clinical diagnosis, treatment decisions, or any
> other direct patient-care purpose without independent clinical validation
> and the appropriate regulatory approval. Use it for research, education,
> and building/prototyping workflows -- not as a substitute for professional
> medical judgment.

## Why MIAI?

Modern medical imaging research requires combining many disconnected tools:
DICOM management, NIfTI conversion, preprocessing, dataset organization, deep
learning frameworks, evaluation, visualization, and experiment management.
Excellent libraries exist individually, but there is no unified framework that
connects the complete research workflow. MIAI fills this gap — not as another
deep learning library, but as a **workflow framework** that standardizes
everything surrounding deep learning.

## Documentation

| Document | Description |
|---|---|
| [docs/vision.md](docs/vision.md) | Mission, philosophy, target users, scope |
| [docs/architecture.md](docs/architecture.md) | Ecosystem layout and package boundaries |
| [docs/roadmap.md](docs/roadmap.md) | Phased development plan |
| [docs/coding_standards.md](docs/coding_standards.md) | Style, testing, versioning conventions |
| [docs/api_design.md](docs/api_design.md) | API design principles across packages |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |

## Ecosystem

```
MIAI
├── miai-core               # utilities, config, logging, IO, exceptions        [done]
├── miai-dicom              # DICOM read/write, anonymization, series loading   [done]
├── miai-pipeline           # end-to-end clinical workflow orchestration        [done]
├── miai-transforms         # image preprocessing / augmentation                [done]
├── miai-datasets           # dataset management                                [done]
├── miai-segmentation       # segmentation models & training                    [done]
├── miai-registration       # image registration                                [done]
├── miai-reconstruction     # image reconstruction                              [done]
├── miai-diffusion          # generative / diffusion models                     [done]
├── miai-foundation-models  # pretrained-model embeddings / feature extraction  [done]
├── miai-evaluation         # metrics and evaluation                            [done]
├── miai-visualization      # visualization tooling                             [done]
├── miai-deploy             # clinical deployment                               [done]
└── miai-examples           # end-to-end examples                               [done]
```

All packages above live in this single monorepo, each under its own
`src/<package>` namespace with its own tests -- see
[docs/architecture.md](docs/architecture.md) for why, and why no package has
been split into its own repository yet.

## Installation

`miai-core` is not yet published to PyPI. Install from source:

```bash
git clone https://github.com/juancis44/MIAI.git
cd MIAI
pip install -e ".[dev]"
```

Quick example — `miai-core` configuration and logging:

```python
from miai_core import MIAIBaseConfig, get_logger

class TrainingConfig(MIAIBaseConfig):
    learning_rate: float
    batch_size: int

config = TrainingConfig.from_yaml("configs/train.yaml")
logger = get_logger(__name__)
logger.info("Loaded config: %s", config)
```

Quick example — `miai-dicom` reading, metadata, and de-identification:

```python
from miai_dicom import read_dicom, extract_metadata, anonymize, write_dicom

dataset = read_dicom("scan.dcm")
metadata = extract_metadata(dataset)
deidentified = anonymize(dataset)
write_dicom(deidentified, "scan_anonymized.dcm")
```

Quick example — `miai-pipeline` config-driven orchestration:

```python
from miai_pipeline import Pipeline, PipelineConfig, PipelineContext

config = PipelineConfig.from_yaml("configs/pipeline.yaml")
pipeline = Pipeline.from_config(config)

context = PipelineContext()
context.set("dicom_dir", "data/raw_dicom")
result = pipeline.run(context)
print(result.require("manifest_path"))
```

The same pipeline can also be run from the command line, without writing
any Python, via the `miai-pipeline` console script installed alongside the
package:

```bash
miai-pipeline validate configs/pipeline.yaml      # check the config, don't run it
miai-pipeline list-stages                         # list every registered stage type
miai-pipeline run configs/pipeline.yaml --set dicom_dir=data/raw_dicom
```

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
