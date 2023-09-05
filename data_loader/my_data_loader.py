import os

import transformers  # type: ignore [import]
from torch.utils.data import DataLoader
from typing import (Any, )

from .my_dataset import MyDataset
from tokenizer import Tokenizer

# NOTE: https://huggingface.co/docs/transformers/model_doc/t5#training
PREFIX = "summarize: "


def get_data_loader(
    path: os.PathLike,
    source_tokenizer: transformers.PreTrainedTokenizerBase,
    target_tokenizer: Tokenizer,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int
) -> DataLoader:

    def collate_fn(
        samples: list[dict[str, Any]]
    ) -> dict[str, transformers.tokenization_utils_base.BatchEncoding]:

        sources = [PREFIX + sample['specification'] for sample in samples]
        targets = [sample['stringified'] for sample in samples]
        names = [sample['name'] for sample in samples]

        source_encodings = source_tokenizer.batch_encode_plus(
            sources,
            add_special_tokens=False,
            max_length=512,
            padding=True,
            return_tensors='pt',
            truncation=True,
        )
        target_encodings = target_tokenizer.batch_encode_plus(
            targets,
            padding=True,
            add_special_tokens=False,
            return_tensors='pt'
        )

        return {
            'names': names,
            'sources_origin': sources,
            'targets_origin': targets,
            'sources': source_encodings,
            'targets': target_encodings,
        }

    dataset = MyDataset(path)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    return data_loader
