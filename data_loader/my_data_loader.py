import jsonlines
import torch
import pandas as pd
from transformers import PreTrainedTokenizerBase  # type: ignore [import]
from torch.utils.data import DataLoader

from .my_dataset import MyDataset


def _get_data_frame(
    file: str,
    tokenizer: PreTrainedTokenizerBase,
    separator: str = ' // ',
    subseparator: str = ' / '
) -> pd.DataFrame:

    sources = []
    targets = []
    indexs = []
    names = []

    with jsonlines.open(file) as f:
        for idx, obj in enumerate(f):

            source = obj['description']['spec']
            if source == '':
                continue

            indexs.append(obj['name']['index'])
            names.append(obj['name']['name'])
            sources.append(source)

            target1 = subseparator.join(obj['spec']['grammer'])
            target2 = subseparator.join(obj['spec']['constraints'])

            target1 = target1.replace("<S>", "<R>").replace("<s>", "<p>")
            target = target1 + ' // ' + target2
            targets.append(target + tokenizer.eos_token)

    df = pd.DataFrame()
    df['index'] = indexs
    df['name'] = names
    df['source'] = sources
    df['target'] = targets

    return df


def get_data_loader(
    file: str,
    tokenizer: torch.nn.Module,
    separator: str = ' / ',
    subseparator: str = ' // ',
    *,
    batch_size: int,
    shuffle: bool,
    source_len: int,
    target_len: int,
    source_title: str,
    target_title: str,
    num_workers: int
) -> DataLoader:

    df = _get_data_frame(file, tokenizer)
    dataset = MyDataset(
        df, tokenizer, source_len, target_len, source_title, target_title
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return data_loader
