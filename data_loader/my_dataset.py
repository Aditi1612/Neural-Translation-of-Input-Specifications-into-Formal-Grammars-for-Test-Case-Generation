from typing import (Any, )

import torch
import transformers
import pandas as pd

from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizerBase,
        source_len: int,
        target_len: int,
        source_title: str,
        target_title: str
    ) -> None:
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_title (str): column name of source text
            target_title (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe

        self.source_len = source_len
        self.target_len = source_len

        self.inputs = self.data[source_title]
        self.targets = self.data[target_title]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index) -> dict[str, Any]:
        """return the input ids, attention masks and target ids"""

        source_text = str(self.inputs[index])
        target_text = str(self.targets[index])

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.target_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }
