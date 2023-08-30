from typing import (Optional, )
import random

import torch
import transformers  # type: ignore [import]

from base.base_trainer import BaseTrainer
from tokenizer import Tokenizer


class T5Trainer(BaseTrainer):
    """ See the following guideline
    https://huggingface.co/docs/transformers/model_doc/t5#training
    """
    ignore_index = -100

    def __init__(
        self,
        model: transformers.T5ForConditionalGeneration,
        optimizer: torch.optim.Optimizer,
        device: str,
        data_loader: torch.utils.data.DataLoader,
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer,
        valid_data_loader: Optional[torch.utils.data.DataLoader] = None,
        lr_scheduler: None = None,
        *,
        epochs: int,
        save_dir: str,
        save_period: int,
        valid_period: int,
        length_penalty: float = 1.0,
        max_new_tokens: int = 150,
        num_beams: int = 10,
        repetition_penalty: float = 2.5,
    ) -> None:
        super().__init__(
            model, optimizer,
            epochs=epochs, save_dir=save_dir, save_period=save_period
        )
        self.model: transformers.T5ForConditionalGeneration = self.model
        self.device = device
        self.data_loader = data_loader
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.valid_data_loader = valid_data_loader
        self.do_validation = valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.valid_period = valid_period

        self.length_penalty = length_penalty
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.repetition_penalty = repetition_penalty

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        for batch_idx, batch in enumerate(self.data_loader):

            sources = batch['sources']
            targets = batch['targets']

            input_ids = sources.input_ids.to(self.device)
            attention_mask = sources.attention_mask.to(self.device)
            labels = targets.input_ids.to(self.device)

            pad_token_id = self.target_tokenizer.pad_token_id
            pad_token_indexes = (labels == pad_token_id)
            labels[pad_token_indexes] = self.ignore_index

            self.optimizer.zero_grad()
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = output.loss

            loss.backward()
            self.optimizer.step()

            # TODO: Use logger
            epoch_progress = batch_idx / len(self.data_loader)
            log_message = "Train epoch: {} {:.2f}% Loss: {:.6f}".format(
                    epoch, epoch_progress * 100, loss.item())
            print(log_message)

        if self.do_validation and epoch % self.valid_period == 0:
            self._valid_epoch(epoch)

    def _valid_epoch(self, epoch: int) -> None:

        assert self.valid_data_loader is not None

        losses = []
        generated_decodings = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data_loader):

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
                losses.append(output.loss)

                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    early_stopping=True,
                    length_penalty=self.length_penalty,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    repetition_penalty=self.repetition_penalty
                )

                generated_batch_decodings = (
                    self.target_tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True)
                )
                generated_decodings.extend(generated_batch_decodings)

        avg_loss = sum(losses) / len(losses)
        sampled_decoding = random.choice(generated_decodings)

        # XXX: Use logger
        print(f"Avg. loss: {avg_loss}")
        print(f"Sampled decoding: {sampled_decoding}")
