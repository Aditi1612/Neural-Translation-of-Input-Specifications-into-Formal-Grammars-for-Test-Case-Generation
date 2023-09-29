import json
import random
from pathlib import Path
from typing import (Any, )

import torch
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer  # type: ignore [import]

from data_loader import MyDataset
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer


# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)  # numpy random seed
random.seed(SEED)  # python random seed


def main(config: dict[str, Any]) -> None:

    data_dir = Path(config['data_dir'])
    train_data_path = data_dir / config['train_data']
    pretrained_model_name = config['pretrained']

    # Create a tokenizer for normalization
    source_tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
    target_tokenizer = CcfgTokenizer(source_tokenizer)

    def stringified_to_grammar(stringified: str):
        production_encoding, constraint_encoding = (
            target_tokenizer.encode_to_splited(stringified))

        subseparator = target_tokenizer.subseparator
        production_decoding = target_tokenizer.decode(production_encoding)
        constraint_decoding = target_tokenizer.decode(constraint_encoding)

        productions = production_decoding.split(subseparator)
        productions = list(map(str.strip, productions))
        constraints = constraint_decoding.split(subseparator)
        constraints = list(map(str.strip, constraints))

        grammar = {'productions': productions, 'constraints': constraints}
        return grammar

    train_dataset = MyDataset(train_data_path)
    production_lengths = []
    constraint_lengths = []
    for data in tqdm(train_dataset):

        stringified = data['stringified']

        production_encoding, constraint_encoding = (
            target_tokenizer.encode_to_splited(stringified))
        subseparator = target_tokenizer.subseparator
        production_decoding = target_tokenizer.decode(production_encoding)
        constraint_decoding = target_tokenizer.decode(constraint_encoding)

        num_production_tokens = len(production_encoding)
        num_productions = len(production_decoding.split(subseparator))
        avg_production_len = num_production_tokens / num_productions

        num_constraint_tokens = len(constraint_encoding)
        num_constraints = len(constraint_decoding.split(subseparator))
        avg_constraint_len = num_constraint_tokens / num_constraints

        production_lengths.append(avg_production_len)
        constraint_lengths.append(avg_constraint_len)

    print("Average number of tokens per production:")
    print(sum(production_lengths)/len(production_lengths))
    print(sum(constraint_lengths)/len(constraint_lengths))


if __name__ == "__main__":
    with open('./config.json') as fp:
        config = json.load(fp)
    main(config)
