import os

import torch
import transformers  # type: ignore [import]
from torch.utils.data import DataLoader
from typing import (Any, cast, )

from .my_dataset import MyDataset
from tokenizer import Tokenizer
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer

# NOTE: https://huggingface.co/docs/transformers/model_doc/t5#training
PREFIX = "summarize: "


def get_my_data_loader(
    path: os.PathLike,
    source_tokenizer: transformers.PreTrainedTokenizerBase,
    target_tokenizer: CcfgTokenizer,
    source_encoding_args: dict[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int
) -> DataLoader:

    def collate_fn(
        samples: list[dict[str, Any]]
    ) -> dict[str, Any]:
        # names = [sample['name'] for sample in samples]
        sources = [PREFIX + sample['specification'] for sample in samples]
        source_encodings = source_tokenizer.batch_encode_plus(
            sources, **source_encoding_args)
        stringifieds = [cast(str, sample['stringified']) for sample in samples]
        production_encodings, constraint_encodings = (
            target_tokenizer.batch_encode_to_splited(stringifieds))

        return {
            'samples': samples,
            'input_ids': source_encodings.input_ids,
            'attention_mask': source_encodings.attention_mask,
            'productions': production_encodings,
            'constraints': constraint_encodings,
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


def get_data_loader(
    path: os.PathLike,
    source_tokenizer: transformers.PreTrainedTokenizerBase,
    target_tokenizer: Tokenizer,
    source_encoding_args: dict[str, Any],
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
            sources, **source_encoding_args)
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
