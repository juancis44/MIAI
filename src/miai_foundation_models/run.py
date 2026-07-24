"""Extracting and persisting embeddings for a list of volumes on disk."""

from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk

from miai_core.io import ensure_dir
from miai_core.logging import get_logger
from miai_foundation_models.exceptions import FoundationModelError
from miai_foundation_models.extractor import EmbeddingExtractor, save_embedding

logger = get_logger(__name__)


def extract_embeddings_for_paths(
    extractor: EmbeddingExtractor,
    source_paths: list[str],
    output_dir: str,
) -> list[Path]:
    """Extract and save one embedding per volume in ``source_paths``.

    Reads each volume via SimpleITK (the same I/O convention used
    throughout MIAI -- see :func:`miai_diffusion.denoise.run_denoising`),
    extracts a per-volume embedding with ``extractor``, and saves it as
    a ``.pt`` tensor file next to a name derived from the source file.

    Args:
        extractor: A ready-to-use
            :class:`~miai_foundation_models.extractor.FeatureExtractor`
            (or any object satisfying
            :class:`~miai_foundation_models.extractor.EmbeddingExtractor`).
        source_paths: Volumes to embed.
        output_dir: Directory embeddings are written to (created if
            missing).

    Returns:
        One embedding file path per entry in ``source_paths``, in
        order.

    Raises:
        FoundationModelError: If ``source_paths`` is empty.
    """
    if not source_paths:
        raise FoundationModelError("source_paths is empty; nothing to extract embeddings for.")

    out_dir = ensure_dir(output_dir)
    embedding_paths: list[Path] = []

    for source_path in source_paths:
        image = sitk.ReadImage(str(source_path))
        array = sitk.GetArrayFromImage(image)

        embedding = extractor.extract_volume_embedding(array)

        stem = Path(source_path).name.removesuffix(".nii.gz").removesuffix(".nii")
        out_path = out_dir / f"{stem}_embedding.pt"
        save_embedding(embedding, out_path)
        embedding_paths.append(out_path)
        logger.info("Wrote embedding for %s to %s", source_path, out_path)

    return embedding_paths
