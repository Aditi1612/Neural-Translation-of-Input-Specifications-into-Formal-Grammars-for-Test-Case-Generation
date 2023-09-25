import os
from pathlib import Path
from abc import abstractmethod
from typing import Optional

import torch


class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epochs: int,
        save_dir: str,
        save_period: int,
    ) -> None:

        self.model = model
        self.optimizer = optimizer

        self.epochs = epochs
        self.start_epoch = 1
        self.checkpoint_dir = Path(save_dir)
        self.save_period = save_period

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict[str, float]:
        raise NotImplementedError

    def train(self) -> list[dict[str, float]]:
        losses: list[dict[str, float]] = []
        for epoch in range(self.start_epoch, self.epochs + 1):
            loss_dictionary = self._train_epoch(epoch)
            losses.append(loss_dictionary)

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
        return losses

    def _save_checkpoint(
        self, epoch: int, basename: Optional[str] = None
    ) -> None:
        arch = type(self.model).__name__
        optimizer_type = type(self.optimizer).__name__

        state = {
            'arch': arch,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_type': optimizer_type,
        }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if basename is None:
            basename = f'checkpoint-epoch{epoch}.pth'
        filename = self.checkpoint_dir / basename
        torch.save(state, filename)

    def _resume_checkpoint(self, resume_path: os.PathLike) -> None:
        checkpoint = torch.load(resume_path)

        self.start_epoch = checkpoint['epoch'] + 1
        checkpoint_arch = checkpoint['arch']
        arch = type(self.model).__name__
        if checkpoint_arch != arch:
            raise ValueError(
                "Trainer's model architecture differs from checkpoint's.")
        self.model.load_state_dict(checkpoint['state_dict'])

        checkpoint_optimizer_type = checkpoint['optimizer_type']
        optimizer_type = type(self.optimizer).__name__
        if checkpoint_optimizer_type != optimizer_type:
            raise ValueError("Trainer's optimizer differs from checkpoint's.")
        self.optimizer.load_state_dict(checkpoint['optimizer'])
