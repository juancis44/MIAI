# MIAI — Medical Image Artificial Intelligence

**An open-source ecosystem for reproducible medical imaging workflows.**

MIAI simplifies the development of reproducible, modular, and clinically-oriented
medical imaging workflows. Rather than replacing existing libraries such as
[MONAI](https://monai.io/), [PyTorch](https://pytorch.org/), or
[SimpleITK](https://simpleitk.org/), MIAI integrates them into a coherent
software architecture focused on research reproducibility, software
engineering best practices, and clinical applicability.

> **Status:** Phase 3 — `miai-core`, `miai-dicom`, and `miai-pipeline`
> implemented. See [docs/roadmap.md](docs/roadmap.md) for what's next.

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

## Ecosystem (planned)

```
MIAI
├── miai-core            # utilities, config, logging, IO, exceptions
├── miai-dicom            # DICOM read/write, anonymization, series loading
├── miai-pipeline          # end-to-end clinical workflow orchestration
├── miai-transforms        # image preprocessing / augmentation
├── miai-datasets          # dataset management
├── miai-segmentation      # segmentation models & training
├── miai-registration      # image registration
├── miai-reconstruction    # image reconstruction
├── miai-diffusion         # generative / diffusion models
├── miai-evaluation        # metrics and evaluation
├── miai-visualization     # visualization tooling
├── miai-deploy            # clinical deployment
└── miai-examples          # end-to-end examples
```

During Phase 0–1, this repository holds the project design and the first
package, `miai-core`, as a single monorepo. Packages may be split into
separate repositories later as the ecosystem matures.

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

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
