import torch

from base.base_trainer import BaseTrainer


class T5Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer,
        device,
        data_loader,
        tokenizer,
        valid_data_loader=None,
        lr_scheduler=None,
        *,
        epochs: int,
        save_dir: str,
        save_period: int,
    ) -> None:
        super().__init__(
            model, optimizer,
            epochs=epochs, save_dir=save_dir, save_period=save_period
        )
        self.device = device
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.valid_data_loader = valid_data_loader
        self.do_validation = valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.log_step = 10

    def _train_epoch(self, epoch: int) -> None:
        self.model.train()
        for batch_idx, data in enumerate(self.data_loader):

            # XXX: I copied the below code without brain
            y = data["target_ids"].to(self.device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
            ids = data["source_ids"].to(self.device, dtype=torch.long)
            mask = data["source_mask"].to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            output = self.model(
                input_ids=ids,
                attention_mask=mask,
                decoder_input_ids=y_ids,
                labels=lm_labels,
            )
            loss = output[0]
            loss.backward()
            self.optimizer.step()

            # TODO: Use logger
        print(f"Train epoch: {epoch} Loss: {loss.item():.6f}")

        if self.do_validation:
            self._valid_epoch(epoch)

    def _valid_epoch(self, epoch: int) -> None:
        predictions = []
        actuals = []
        sources = []
        losses = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                y = data['target_ids'].to(self.device, dtype=torch.long)
                ids = data['source_ids'].to(self.device, dtype=torch.long)
                mask = data['source_mask'].to(self.device, dtype=torch.long)

                num_of_beam_sample = 10  # XXX

                generated_ids = self.model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=10,  # XXX: max_target_text_length
                    num_beams=10,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True,
                    num_return_sequences=num_of_beam_sample
                )
                losses.append(self.model(input_ids=ids, labels=y).loss)

                def tokenizer_decode(e):
                    return self.tokenizer.decode(
                        e,
                        skip_special_tokens=True,  # XXX
                        clean_up_tokenization_spaces=True
                    )

                preds = [tokenizer_decode(g) for g in generated_ids]
                target = [tokenizer_decode(t) for t in y]
                source = [tokenizer_decode(i) for i in ids]

                sources.extend(source)
                actuals.extend(target)
                num_of_problem = len(preds) // num_of_beam_sample
                result = []
                for i in range(num_of_problem):
                    result.append(preds[:num_of_beam_sample])
                    del preds[:num_of_beam_sample]
                predictions.extend(result)
        # XXX:
        print(losses)
