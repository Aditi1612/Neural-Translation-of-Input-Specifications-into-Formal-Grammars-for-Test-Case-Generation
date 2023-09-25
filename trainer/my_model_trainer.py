import logging
import random
import copy
import math
from collections import deque
from typing import (Optional, Protocol, Callable, Any, )

import torch
from torch.utils.data import DataLoader

from base.base_trainer import BaseTrainer
from data_loader import MyDataset
from model import MyModel


logger = logging.getLogger(__name__)


def _average(a: list[float]) -> float:
    if len(a) == 0:
        return 0
    return sum(a) / len(a)


class PseudoLabeler(Protocol):
    def __call__(
        self, unlabeled_data: dict[str, Any], model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        pass


class MyModelTrainer(BaseTrainer):
    """ See the following guideline
    https://huggingface.co/docs/transformers/model_doc/t5#training
    """
    ignore_index = -100

    def __init__(
        self,
        model: MyModel,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        data_loader: DataLoader,
        # valid_data_loader: Optional[DataLoader] = None,
        validate: Optional[Callable[[MyModel], float]] = None,
        unlabeled_data_list: list[dict[str, Any]] = [],
        pseudo_labeler: PseudoLabeler = lambda x, y: None,
        lr_scheduler: None = None,
        *,
        epochs: int,
        save_dir: str,
        save_period: int,
        valid_period: int = -1,
        pseudo_label_period: int = -1,
        length_penalty: float = 1.0,
        max_new_tokens: int = 150,
        num_beams: int = 10,
        repetition_penalty: float = 2.5,
        pseudo_label_samples: int = 100,
        early_stopping_patience: int = 5,
    ) -> None:
        super().__init__(
            model, optimizer,
            epochs=epochs, save_dir=save_dir, save_period=save_period
        )
        self.model: MyModel = model
        self.device = device
        self.data_loader = data_loader
        # self.valid_data_loader = valid_data_loader
        self._validate_callback = validate
        unlabeled_data_list = copy.deepcopy(unlabeled_data_list)
        random.shuffle(unlabeled_data_list)
        self.unlabeled_data_deque = deque(unlabeled_data_list)
        self.pseudo_labeler = pseudo_labeler
        self.lr_scheduler = lr_scheduler

        self.valid_period = valid_period
        self.pseudo_label_period = pseudo_label_period
        self.length_penalty = length_penalty
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty

        self.source_tokenizer = model.source_tokenizer
        self.target_tokenizer = model.target_tokenizer
        self.do_validation = validate is not None
        self.do_pseudo_labeling = len(unlabeled_data_list) > 0
        self.pseudo_labeled_dataset = None
        self.pseudo_labeled_data_loader: Optional[DataLoader] = None
        if self.do_pseudo_labeling:
            self.pseudo_labeled_dataset = MyDataset()
        self.pseudo_label_samples = pseudo_label_samples
        self.early_stopping_patience = early_stopping_patience
        self.do_early_stopping = (
            early_stopping_patience > 0 and self.do_validation)

    def train(self) -> list[dict[str, float]]:
        losses: list[dict[str, float]] = []
        patient_count = -1
        best_correctness = 0.0
        for epoch in range(self.start_epoch, self.epochs + 1):
            loss_dictionary = self._train_epoch(epoch)
            losses.append(loss_dictionary)

            valid_correctness = loss_dictionary.get('valid', None)
            if valid_correctness is not None:
                if (
                    self.do_early_stopping
                    and (valid_correctness < best_correctness)
                ):
                    patient_count += 1
                    if patient_count >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        self._save_checkpoint(
                            epoch,
                            basename=f'early-stop-checkpoint-epoch{epoch}.pth'
                        )
                        break
                else:
                    patient_count = 0
                    best_correctness = valid_correctness
                    logger.info(
                        "Best correctness at epoch {}: {:.6f}%"
                        .format(epoch, best_correctness * 100)
                    )
                    self._save_checkpoint(
                        epoch,
                        basename=f'best-checkpoint-epoch{epoch}.pth'
                    )

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
        return losses

    def _get_batch_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        pad_token_id = self.target_tokenizer.pad_token_id

        production_labels = batch['productions'].to(self.device)
        production_pad_token_indexes = (production_labels == pad_token_id)
        production_labels[production_pad_token_indexes] = self.ignore_index

        constraint_labels = batch['constraints'].to(self.device)
        constraint_pad_token_indexes = (constraint_labels == pad_token_id)
        constraint_labels[constraint_pad_token_indexes] = self.ignore_index

        production_output, constraint_output = self.model(
            input_ids, attention_mask, production_labels, constraint_labels)
        loss = production_output.loss + constraint_output.loss
        return loss

    def _train_batch(self, batch: dict[str, torch.Tensor]) -> float:
        loss = self._get_batch_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        loss_dictionary: dict[str, float] = {}
        self.model.train()

        total_data_loader_len = len(self.data_loader)
        if self.pseudo_labeled_data_loader is not None:
            total_data_loader_len += len(self.pseudo_labeled_data_loader)

        labeled_data_losses: list[float] = []
        for batch_index, batch in enumerate(self.data_loader):
            loss = self._train_batch(batch)
            labeled_data_losses.append(loss)

            epoch_progress = batch_index / total_data_loader_len
            logger.debug(
                "Train epoch: {} {:.2f}% Loss: {:.6f}"
                .format(epoch, epoch_progress * 100, loss)
            )

        pseudo_labeled_data_losses: list[float] = []
        if self.pseudo_labeled_data_loader is not None:
            for batch_index, batch in enumerate(
                self.pseudo_labeled_data_loader,
                start=len(self.data_loader)
            ):
                loss = self._train_batch(batch)
                pseudo_labeled_data_losses.append(loss)

                epoch_progress = batch_index / total_data_loader_len
                logger.debug(
                    "Train epoch: {} {:.2f}% Loss: {:.6f}"
                    .format(epoch, epoch_progress * 100, loss)
                )

        average_labeled_loss = _average(labeled_data_losses)
        loss_dictionary['labeled'] = average_labeled_loss
        logger.info(
            f"Train epoch: {epoch} "
            + "Avg. labeled loss: {:.6f}".format(average_labeled_loss))

        if self.do_pseudo_labeling:
            average_pseudo_labeled_loss = _average(pseudo_labeled_data_losses)
            logger.info(
                "Train epoch: {} Avg. pseudo-labeled loss: {:.6f}"
                .format(epoch, average_pseudo_labeled_loss)
            )
            loss_dictionary['pseudo_labeled'] = average_pseudo_labeled_loss

        if self.do_validation and epoch % self.valid_period == 0:
            average_valid_loss = self._valid_epoch(epoch)
            loss_dictionary['valid'] = average_valid_loss

        if self.do_pseudo_labeling and epoch % self.pseudo_label_period == 0:
            self._pseudo_label(epoch)

        return loss_dictionary

    def _valid_epoch(self, epoch: int) -> float:

        # assert self.valid_data_loader is not None
        assert self._validate_callback is not None
        valid_correctness = self._validate_callback(self.model)
        return valid_correctness

    def _pseudo_label(self, epoch: int) -> None:

        assert self.pseudo_labeled_dataset is not None

        pseudo_labeled_data_list = []
        failed_data_list = []

        pseudo_label_samples = min(
            len(self.unlabeled_data_deque), self.pseudo_label_samples)

        logging_frequency = int(math.sqrt(pseudo_label_samples))
        for index in range(pseudo_label_samples):
            if index % logging_frequency == 0:
                logger.debug(
                    "Pseudo-labeling: {} {:.2f}%"
                    .format(epoch, 100 * index/pseudo_label_samples)
                )
            unlabeled_data = self.unlabeled_data_deque.popleft()
            description = unlabeled_data['description']
            specification = MyDataset.get_specification(description)
            pseudo_label = self.pseudo_labeler(unlabeled_data, self.model)
            if pseudo_label is None:
                failed_data_list.append(unlabeled_data)
                continue
            unlabeled_data['grammar'] = pseudo_label
            unlabeled_data['specification'] = specification
            pseudo_labeled_data_list.append(unlabeled_data)

        self.unlabeled_data_deque.extend(failed_data_list)

        logger.info(
            "Pseudo-labeled entries: {}(+{})"
            .format(
                len(self.pseudo_labeled_dataset),
                len(pseudo_labeled_data_list)
            )
        )
        self.pseudo_labeled_dataset.extend(pseudo_labeled_data_list)

        should_create_pseudo_labeled_data_loader = (
            self.pseudo_labeled_data_loader is None
            and len(self.pseudo_labeled_dataset) > 0
        )
        if should_create_pseudo_labeled_data_loader:
            self.pseudo_labeled_data_loader = DataLoader(
                self.pseudo_labeled_dataset,
                batch_size=self.data_loader.batch_size,
                shuffle=True,
                num_workers=self.data_loader.num_workers,
                collate_fn=self.data_loader.collate_fn,
            )
