import os
import copy
from typing import (Any, Optional, cast, )

import jsonlines

from torch.utils.data import Dataset
from tokenizer import CountingContextFreeGrammarTokenizer as CCFGTokenizer


class MyDataset(Dataset):
    def __init__(self, path: Optional[os.PathLike] = None) -> None:
        if path is None:
            self.data: list[dict[str, Any]] = []
            return

        with jsonlines.open(path, 'r') as f:
            self.data = cast(
                list[dict[str, Any]],
                list(map(MyDataset.preprocess, f))
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, str]:
        """return the input ids, attention masks and target ids"""
        return self.data[index]

    def extend(self, dataset: list[dict[str, Any]]) -> None:
        self.data.extend(map(MyDataset.preprocess, dataset))

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
    def partial_stringify(productions_or_constraints: list[str]) -> str:
        data = cast(list[str], productions_or_constraints)
        return f" {CCFGTokenizer.subseparator} ".join(data)

    @staticmethod
    def stringify(grammar: dict[str, list[str]]) -> str:
        productions = cast(list[str], grammar['productions'])
        constraints = cast(list[str], grammar['constraints'])
        return f" {CCFGTokenizer.separator} ".join([
            MyDataset.partial_stringify(productions),
            MyDataset.partial_stringify(constraints)
        ])

    @staticmethod
    def preprocess(obj: dict[str, Any]) -> dict[str, Any]:
        obj = copy.deepcopy(obj)

        description = obj['description']
        grammar = obj['grammar']
        obj['specification'] = MyDataset.get_spec(description)
        obj['stringified'] = MyDataset.stringify(grammar)
        return obj
