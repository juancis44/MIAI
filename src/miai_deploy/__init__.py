"""miai-deploy: portable export and bundling of trained models.

MIAI's reference deployment task is *portable export*, not live model
serving: converting a trained model into a self-contained artifact
(TorchScript or ONNX) that other systems can load without depending on
MIAI or the original Python model-construction code, packaged together
with reproducibility metadata (see :class:`~miai_deploy.bundle.BundleMetadata`).

See :func:`~miai_deploy.export.export_model` for the export step alone,
and :func:`~miai_deploy.bundle.write_bundle` for exporting plus writing
metadata as a single bundle directory.
"""

from __future__ import annotations

from miai_deploy.bundle import BundleMetadata, write_bundle
from miai_deploy.exceptions import DeployError
from miai_deploy.export import ExportConfig, export_model

__version__ = "0.1.0"

__all__ = [
    "DeployError",
    "ExportConfig",
    "export_model",
    "BundleMetadata",
    "write_bundle",
]
