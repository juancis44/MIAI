# MIAI — Medical Image Artificial Intelligence

**An open-source ecosystem for reproducible medical imaging workflows.**

MIAI simplifies the development of reproducible, modular, and clinically-oriented
medical imaging workflows. Rather than replacing existing libraries such as
[MONAI](https://monai.io/), [PyTorch](https://pytorch.org/), or
[SimpleITK](https://simpleitk.org/), MIAI integrates them into a coherent
software architecture focused on research reproducibility, software
engineering best practices, and clinical applicability.

> **Status:** Phase 1 — `miai-core` implemented (config, logging, IO,
> exceptions, typing, utilities). See [docs/roadmap.md](docs/roadmap.md)
> for what's next.

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

Quick example:

```python
from miai_core import MIAIBaseConfig, get_logger

class TrainingConfig(MIAIBaseConfig):
    learning_rate: float
    batch_size: int

config = TrainingConfig.from_yaml("configs/train.yaml")
logger = get_logger(__name__)
logger.info("Loaded config: %s", config)
```

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).
