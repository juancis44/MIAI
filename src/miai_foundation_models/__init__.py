"""miai-foundation-models: pretrained-model feature extraction for volumes.

Wraps a pretrained Hugging Face Hub vision model (2D) as a per-volume
(3D) embedding extractor, using a "2.5D" slice-and-aggregate strategy.
Intended as a lightweight alternative to training a task-specific model
from scratch: the resulting embeddings can feed downstream retrieval,
clustering, or simple classification workflows without any fine-tuning.

See :class:`~miai_foundation_models.extractor.FeatureExtractor` for the
core API, and :func:`~miai_foundation_models.run.extract_embeddings_for_paths`
for extracting and persisting embeddings for a list of volumes on disk.
"""

from __future__ import annotations

from miai_foundation_models.exceptions import FoundationModelError
from miai_foundation_models.extractor import (
    EmbeddingExtractor,
    FeatureExtractor,
    FeatureExtractorConfig,
    save_embedding,
)
from miai_foundation_models.run import extract_embeddings_for_paths

__version__ = "0.1.0"

__all__ = [
    "FoundationModelError",
    "EmbeddingExtractor",
    "FeatureExtractor",
    "FeatureExtractorConfig",
    "save_embedding",
    "extract_embeddings_for_paths",
]
