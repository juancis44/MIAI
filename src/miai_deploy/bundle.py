"""Packaging an exported model with reproducibility metadata.

A "bundle" is a self-contained directory: the exported model artifact
(see :mod:`miai_deploy.export`) plus a ``metadata.yaml`` describing
what it is, so an exported model is never handed off without knowing
which model/version/config produced it -- in the spirit of MIAI's
reproducibility-first design (see docs/vision.md).
"""

from __future__ import annotations

from pathlib import Path

import torch

from miai_core.config import MIAIBaseConfig
from miai_core.io import ensure_dir
from miai_core.logging import get_logger
from miai_deploy.export import ExportConfig, export_model

logger = get_logger(__name__)

_EXTENSION_BY_FORMAT = {"torchscript": "pt", "onnx": "onnx"}


class BundleMetadata(MIAIBaseConfig):
    """Reproducibility metadata written alongside an exported model.

    Attributes:
        name: A short identifier for the model, e.g. ``"miai-unet-liver"``.
        version: A version string for this specific export, e.g.
            ``"1.0.0"`` or a commit hash.
        description: Free-text description (task, training data,
            intended use).
        extra: Arbitrary additional key-value metadata, e.g.
            ``{"dice_val": "0.91", "source_checkpoint": "best_model.pt"}``.
    """

    name: str
    version: str
    description: str = ""
    extra: dict[str, str] = {}


def write_bundle(
    model: torch.nn.Module,
    checkpoint_path: str,
    export_config: ExportConfig,
    metadata: BundleMetadata,
    output_dir: str,
) -> Path:
    """Export ``model`` and write it, with metadata, as a bundle directory.

    Args:
        model: An untrained model with the same architecture used to
            produce ``checkpoint_path``.
        checkpoint_path: Path to a trained checkpoint.
        export_config: Export format/tracing parameters.
        metadata: Reproducibility metadata for this bundle.
        output_dir: Directory the bundle is written to (created if
            missing). Contains ``model.<ext>`` and ``metadata.yaml``.

    Returns:
        ``output_dir`` as a :class:`pathlib.Path`.
    """
    out_dir = ensure_dir(output_dir)
    extension = _EXTENSION_BY_FORMAT[export_config.format]

    export_model(model, checkpoint_path, export_config, str(out_dir / f"model.{extension}"))
    metadata.to_yaml(out_dir / "metadata.yaml")

    logger.info(
        "Wrote deployment bundle '%s' (v%s) to %s", metadata.name, metadata.version, out_dir
    )
    return out_dir
