import logging
import random
import copy
from typing import (Optional, Any, cast, )

import torch
import transformers  # type: ignore [import]

# from base.base_trainer import BaseTrainer
from tokenizer import Tokenizer
from data_loader import MyDataset
from .t5_trainer import T5Trainer


class T5SelfTrainingTrainer(T5Trainer):
    def __init__(
        self,
        model: transformers.T5ForConditionalGeneration,
        optimizer: torch.optim.Optimizer,
        device: str,
        labeled_data_loader: torch.utils.data.DataLoader,
        unlabeled_data_list: list[dict[str, Any]],
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        valid_data_loader: Optional[torch.utils.data.DataLoader] = None,
        lr_scheduler: None = None,
        *,
        pseudo_label_period: int,
        epochs: int,
        save_dir: str,
        save_period: int,
        valid_period: int = -1,
        length_penalty: float = 1.0,
        max_new_tokens: int = 150,
        num_beams: int = 10,
        repetition_penalty: float = 2.5,
    ) -> None:
        super().__init__(
            model, optimizer, device,
            labeled_data_loader,
            source_tokenizer, target_tokenizer,
            valid_data_loader, lr_scheduler,
            epochs=epochs,
            save_dir=save_dir,
            save_period=save_period,
            valid_period=valid_period,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )

        self.unlabled_data_list = copy.deepcopy(unlabeled_data_list)
        random.shuffle(self.unlabled_data_list)

        self.pseudo_labeled_dataset = MyDataset()
        self.pseudo_labeled_data_loader = torch.utils.data.DataLoader(
            self.pseudo_labeled_dataset,
            batch_size=self.data_loader.batch_size,
            shuffle=True,
            num_workers=self.data_loader.num_workers,
            collate_fn=self.data_loader.collate_fn,
        )
        self.pseudo_label_period = pseudo_label_period

    def _train_batch(self, batch: Any) -> float:
        sources = batch['sources']
        targets = batch['targets']

        input_ids = sources.input_ids.to(self.device)
        attention_mask = sources.attention_mask.to(self.device)
        labels = targets.input_ids.to(self.device)

        pad_token_id = self.target_tokenizer.pad_token_id
        pad_token_indexes = (labels == pad_token_id)
        labels[pad_token_indexes] = self.ignore_index

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = output.loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return cast(float, loss.item())

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()

        total_data_loader_len = (
            len(self.data_loader) + len(self.pseudo_labeled_data_loader))

        labeled_data_losses = []
        for batch_idx, batch in enumerate(self.data_loader):

            loss = self._train_batch(batch)
            labeled_data_losses.append(loss)

            epoch_progress = batch_idx / total_data_loader_len
            log_message = (
                "Train epoch: {} {:.2f}% Loss: {:.6f}"
                .format(epoch, epoch_progress * 100, loss)
            )
            logging.debug(log_message)

        pseudo_labeled_data_losses = []
        for batch_idx, batch in enumerate(self.pseudo_labeled_data_loader):

            pseudo_labeled_data_losses.append(self._train_batch(batch))

            epoch_progress = (
                (batch_idx + len(self.data_loader))
                / total_data_loader_len
            )
            log_message = (
                "Train epoch: {} {:.2f}% Loss: {:.6f}"
                .format(epoch, epoch_progress * 100, loss)
            )
            logging.debug(log_message)

        if len(labeled_data_losses) > 0:
            logging.info(
                "Avg. labeled loss: {:.6f}"
                .format(sum(labeled_data_losses)/len(labeled_data_losses))
            )

        if len(labeled_data_losses) > 0:
            logging.info(
                "Avg. pseudo-labeled loss: {:.6f}"
                .format(
                    sum(pseudo_labeled_data_losses)
                    / len(pseudo_labeled_data_losses)
                )
            )

        if epoch % self.pseudo_label_period == 0:
            self._pseudo_label_epoch(epoch)

    def _pseudo_label_epoch(self, epoch: int) -> None:
        # TODO: self.pseudo_labeled_dataset.extend(pseudo_labeled_data)
        pseudo_labeled_data = []
        logging.info(f"Pseudo labeled entry: {len(pseudo_labeled_data)}")
