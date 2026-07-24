"""Training loop for a MIAI diffusion (noise-prediction) model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch

from miai_core.config import MIAIBaseConfig
from miai_core.io import ensure_dir
from miai_core.logging import get_logger
from miai_diffusion.exceptions import DiffusionError
from miai_diffusion.schedule import NoiseSchedule

logger = get_logger(__name__)


class DiffusionTrainingConfig(MIAIBaseConfig):
    """Configuration for :func:`train_diffusion_model`.

    Attributes:
        max_epochs: Number of training epochs.
        learning_rate: Adam optimizer learning rate.
        device: ``"cpu"`` or ``"cuda"``.
        checkpoint_name: Filename for the checkpoint written under
            ``train_diffusion_model``'s ``checkpoint_dir`` argument.
            Diffusion training has no natural validation metric to pick
            a "best" epoch the way segmentation's Dice does, so the
            final epoch's weights are always the ones checkpointed.
    """

    max_epochs: int = 100
    learning_rate: float = 1e-4
    device: str = "cpu"
    checkpoint_name: str = "diffusion_model.pt"


def train_diffusion_model(
    model: torch.nn.Module,
    train_loader: Iterable[dict[str, torch.Tensor]],
    schedule: NoiseSchedule,
    config: DiffusionTrainingConfig,
    checkpoint_dir: str,
) -> Path:
    """Train a noise-prediction model with the standard DDPM objective.

    For each batch, samples a random timestep and standard normal noise
    per item, forms the noisy input via
    :meth:`~miai_diffusion.schedule.NoiseSchedule.q_sample`, and
    minimizes the mean squared error between the model's predicted
    noise and the actual noise added -- the simplified training
    objective from Ho, Jain & Abbeel 2020 (their Eq. 14).

    Args:
        model: The noise-prediction model to train (e.g. from
            :func:`miai_diffusion.model.build_diffusion_unet`).
        train_loader: Any iterable of batches with an ``"image"`` key
            (typically a :class:`torch.utils.data.DataLoader`, but any
            iterable works -- this function only ever iterates over
            it once per epoch, nothing DataLoader-specific).
        schedule: The noise schedule training samples ``t`` from and
            forms ``x_t`` with. Its tensors should already be on
            ``config.device`` (see the ``device`` argument of
            :class:`~miai_diffusion.schedule.NoiseSchedule`).
        config: Training hyperparameters.
        checkpoint_dir: Directory the checkpoint is written to (created
            if missing).

    Returns:
        Path to the saved checkpoint (a ``torch.save``d state dict).

    Raises:
        DiffusionError: If ``train_loader`` yields no batches.
    """
    device = torch.device(config.device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    out_dir = ensure_dir(checkpoint_dir)
    checkpoint_path = out_dir / config.checkpoint_name

    saw_any_batch = False
    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            saw_any_batch = True
            n_batches += 1
            x0 = batch["image"].to(device)

            t = torch.randint(0, schedule.config.num_timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, t, noise)

            optimizer.zero_grad()
            predicted_noise = model(x_t, t)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if n_batches > 0:
            logger.info(
                "Epoch %d/%d - train loss: %.4f",
                epoch + 1,
                config.max_epochs,
                epoch_loss / n_batches,
            )

    if not saw_any_batch:
        raise DiffusionError("train_loader yielded no batches; cannot train.")

    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path
