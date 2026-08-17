"""Volume feature extraction with a pretrained Hugging Face vision model.

Most Hugging Face vision foundation models (DINOv2, CLIP-style
encoders, etc.) are trained on 2D natural images, not 3D medical
volumes. Rather than requiring a native-3D foundation model (which are
scarce and mostly gated/research-only), :class:`FeatureExtractor` uses
the standard "2.5D" pattern: it runs the 2D model over every slice of a
volume along a chosen axis, then aggregates the per-slice embeddings
into a single per-volume embedding. This keeps the model swappable
(any Hugging Face Hub model ID with a matching image processor works)
without requiring MIAI to ship or train its own 3D encoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import torch

from miai_core.config import MIAIBaseConfig
from miai_core.logging import get_logger
from miai_foundation_models.exceptions import FoundationModelError

logger = get_logger(__name__)


class FeatureExtractorConfig(MIAIBaseConfig):
    """Configuration for :class:`FeatureExtractor`.

    Attributes:
        model_id: A Hugging Face Hub model identifier for a vision
            model, e.g. ``"facebook/dinov2-small"`` (the default -- a
            small, ungated, general-purpose vision transformer; swap
            in a domain-specific encoder, e.g. a published biomedical
            CLIP/DINO variant, by changing this one field).
        device: ``"cpu"`` or ``"cuda"``.
        slice_axis: Which array axis to slice the volume along. Volumes
            follow SimpleITK's ``(D, H, W)`` array convention
            throughout MIAI, so ``0`` (the default) slices along depth.
        token_pooling: How to reduce a single 2D slice's model output
            (a sequence of patch/token embeddings) to one vector.
            ``"cls"`` uses the model's CLS token (``pooler_output`` if
            the model provides one, otherwise the first token of the
            last hidden state); ``"mean"`` mean-pools all tokens.
        slice_pooling: How to reduce the per-slice embeddings of a
            volume to a single per-volume embedding. ``"mean"``
            averages across slices; ``"max"`` takes an element-wise
            maximum, which can better preserve a signal that is only
            present in a few slices.
        batch_size: How many slices to run through the model at once.
    """

    model_id: str = "facebook/dinov2-small"
    device: str = "cpu"
    slice_axis: int = 0
    token_pooling: Literal["cls", "mean"] = "cls"
    slice_pooling: Literal["mean", "max"] = "mean"
    batch_size: int = 8


class _ImageProcessor(Protocol):
    """The subset of a Hugging Face image processor's interface used here.

    Declared as a :class:`typing.Protocol` (rather than importing
    ``transformers``' own type) so tests can inject a lightweight fake
    processor without downloading a real model.
    """

    def __call__(self, images: list[Any], return_tensors: str) -> Any: ...


class EmbeddingExtractor(Protocol):
    """The subset of FeatureExtractor's interface that callers need.

    Declared as a :class:`typing.Protocol` (same rationale as
    :class:`_ImageProcessor` above) so callers -- and tests -- can pass
    any object with a matching ``extract_volume_embedding``, not just a
    real :class:`FeatureExtractor`, without needing to subclass it.
    """

    def extract_volume_embedding(self, volume: npt.NDArray[Any]) -> torch.Tensor:
        """Return a single embedding vector for a 3D volume array."""
        ...


class FeatureExtractor:
    """Extracts a per-volume embedding using a pretrained 2D vision model.

    Constructed either from an already-loaded model/processor pair (for
    testing, or to reuse a model already loaded elsewhere) or via
    :meth:`from_pretrained` to download one from the Hugging Face Hub.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        processor: _ImageProcessor,
        config: FeatureExtractorConfig | None = None,
    ) -> None:
        """Wrap an already-loaded model/processor pair."""
        self.config = config or FeatureExtractorConfig()
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.model.eval()
        self.processor = processor

    @classmethod
    def from_pretrained(cls, config: FeatureExtractorConfig) -> FeatureExtractor:
        """Download and wrap a Hugging Face Hub model.

        Args:
            config: Extraction parameters, including which model to
                load (``config.model_id``).

        Returns:
            A ready-to-use :class:`FeatureExtractor`.
        """
        from transformers import AutoImageProcessor, AutoModel

        logger.info("Downloading foundation model '%s' from Hugging Face Hub.", config.model_id)
        model = cast(torch.nn.Module, AutoModel.from_pretrained(config.model_id))
        processor = cast(_ImageProcessor, AutoImageProcessor.from_pretrained(config.model_id))
        return cls(model, processor, config)

    def _embed_slices(self, slices: list[npt.NDArray[np.float32]]) -> torch.Tensor:
        """Run a batch of 2D grayscale slices through the model.

        Returns one embedding vector per slice, shape
        ``(len(slices), hidden_size)``.
        """
        # Most Hugging Face image processors expect 3-channel images;
        # duplicate the single grayscale channel rather than requiring
        # a 3-channel-aware model.
        rgb_slices = [np.repeat(s[:, :, None], 3, axis=2) for s in slices]

        inputs = self.processor(images=rgb_slices, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.config.token_pooling == "cls":
            pooler_output = getattr(outputs, "pooler_output", None)
            if pooler_output is not None:
                embeddings = pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)

        return cast(torch.Tensor, embeddings)

    def extract_volume_embedding(self, volume: npt.NDArray[Any]) -> torch.Tensor:
        """Extract a single embedding vector for a whole volume.

        Args:
            volume: A grayscale volume array, ``(D, H, W)`` convention.

        Returns:
            A 1D embedding tensor, shape ``(hidden_size,)``.

        Raises:
            FoundationModelError: If ``volume`` has no slices along
                ``config.slice_axis``, or ``config.slice_pooling`` is
                not a recognized value.
        """
        num_slices = volume.shape[self.config.slice_axis]
        if num_slices == 0:
            raise FoundationModelError(
                f"Volume has no slices along axis {self.config.slice_axis}; "
                "cannot extract an embedding from an empty volume."
            )

        slices = [
            np.take(volume, indices=i, axis=self.config.slice_axis).astype(np.float32)
            for i in range(num_slices)
        ]

        slice_embeddings: list[torch.Tensor] = []
        for start in range(0, len(slices), self.config.batch_size):
            batch = slices[start : start + self.config.batch_size]
            slice_embeddings.append(self._embed_slices(batch))
        stacked = torch.cat(slice_embeddings, dim=0)

        if self.config.slice_pooling == "mean":
            return stacked.mean(dim=0)
        if self.config.slice_pooling == "max":
            return stacked.max(dim=0).values
        raise FoundationModelError(f"Unknown slice_pooling strategy: {self.config.slice_pooling!r}")


def save_embedding(embedding: torch.Tensor, path: Path) -> None:
    """Save an embedding tensor to ``path`` (created parents if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embedding.cpu(), path)
