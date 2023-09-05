import os
import copy
from typing import (Any, cast, )

import jsonlines

from torch.utils.data import Dataset
from tokenizer import CountingContextFreeGrammarTokenizer as CCFGTokenizer


class MyDataset(Dataset):
    def __init__(self, path: os.PathLike) -> None:
        with jsonlines.open(path, 'r') as f:
            self.data = cast(
                list[dict[Any, Any]],
                list(map(MyDataset.preprocess, f))
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, str]:
        """return the input ids, attention masks and target ids"""
        return self.data[index]

    @staticmethod
    def get_spec(description: str) -> str:
        description = MyDataset.replace_description(description)
        constraints_start_token = '\nconstraints\n'
        input_start_token = '\ninput\n'
        input_finish_token = '\noutput\n'

        constraints_idx = description.lower().find(constraints_start_token)
        input_start_idx = description.lower().find(input_start_token)
        input_finish_idx = description.lower().find(input_finish_token)

        if input_start_idx < 0 or input_finish_idx < 0:
            return description

        if constraints_idx >= 0:
            return description[constraints_idx:input_finish_idx].strip()
        else:
            return description[input_start_idx:input_finish_idx].strip()

    @staticmethod
    def replace_description(description: str) -> str:
        description_replacements = [
            ('≤', '<='),
            ('\\leq', '<='),
        ]
        for old, new in description_replacements:
            description = description.replace(old, new)
        return description

    @staticmethod
    def stringify(grammar: dict[str, list[str]]) -> str:
        productions = cast(str, grammar['productions'])
        constraints = cast(str, grammar['constraints'])
        return f" {CCFGTokenizer.separator} ".join([
            f" {CCFGTokenizer.subseparator} ".join(productions),
            f" {CCFGTokenizer.subseparator} ".join(constraints)
        ])

    @staticmethod
    def preprocess(obj: dict[str, Any]) -> dict[str, Any]:
        obj = copy.deepcopy(obj)

        description = obj['description']
        grammar = obj['grammar']
        obj['specification'] = MyDataset.get_spec(description)
        obj['stringified'] = MyDataset.stringify(grammar)
        return obj
