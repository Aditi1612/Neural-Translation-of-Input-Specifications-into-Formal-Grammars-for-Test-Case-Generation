import os
from typing import (Any, cast, )

import jsonlines
import pandas as pd

from torch.utils.data import Dataset
from tokenizer import CountingContextFreeGrammarTokenizer as CCFGTokenizer


description_replaces = [
    ('≤', '<='),
    ('\\leq', '<='),
]


class MyDataset(Dataset):

    name_title = "name.name"
    # description_title = "description.description"
    description_title = "spec.spec"
    grammar_title = "spec.grammar"
    constraint_title = "spec.constraints"

    separator = CCFGTokenizer.separator
    subseparator = CCFGTokenizer.subseparator

    def __init__(self, path: os.PathLike) -> None:
        # TODO: Load essential columns only

        data = cast(list[dict[Any, Any]], jsonlines.open(path))
        self.df = pd.json_normalize(data)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, str]:
        """return the input ids, attention masks and target ids"""

        data = self.df.iloc[index]
        name: str = data[self.name_title]
        description: str = data[self.description_title]
        grammar: list[str] = data[self.grammar_title]
        constraint: list[str] = data[self.constraint_title]

        # XXX
        constraint = []

        for old, new in description_replaces:
            description = description.replace(old, new)

        target: str = self.separator.join([
            self.subseparator.join(grammar),
            self.subseparator.join(constraint)
        ])

        return {
            "name": name,
            "source": description,
            "target": target,
        }
