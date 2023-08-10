import os

import transformers
from torch.utils.data import DataLoader

from .my_dataset import MyDataset

PREFIX = "summarize: "


def get_data_loader(
    path: os.PathLike,
    source_tokenizer: transformers.PreTrainedTokenizerBase,
    target_tokenizer: transformers.PreTrainedTokenizerBase,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int
) -> DataLoader:

    def collate_fn(
        samples: list[dict[str, str]]
    ) -> dict[str, transformers.tokenization_utils_base.BatchEncoding]:

        sources = [PREFIX + sample['source'] for sample in samples]
        targets = [sample['target'] for sample in samples]

        source_encodings = source_tokenizer.batch_encode_plus(
            sources,
            add_special_tokens=False,
            max_length=512,
            padding='max_length',
            return_tensors='pt',
            truncation=True,
        )
        target_encodings = target_tokenizer(
            targets,
            padding=True,
            add_special_tokens=False,
            return_tensors='pt'
        )

        return {'sources': source_encodings, 'targets': target_encodings}

    dataset = MyDataset(path)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    return data_loader
