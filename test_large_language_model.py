import argparse
import json
import logging
import random
from pathlib import Path
from typing import (Any, )

import torch
import jsonlines
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
    test_data_path = data_dir / config['test_data']
    pretrained_model_name = config['pretrained']

    # Set variables related to `test_config`
    test_config = config['test']
    logging.info(test_config)
    model_labeled_data_path = Path(test_config['model_labeled_data'])

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

    model_labeled_grammar_dict: dict[str: Any] = {}
    with jsonlines.open(model_labeled_data_path) as model_labeled_data:
        for data in model_labeled_data:
            name = data['name']
            model_labeled_grammar = data['grammar']
            model_labeled_grammar_dict[name] = model_labeled_grammar

    exact_match_productions: list[int] = []
    exact_match_constraints: list[int] = []

    def normalize_list(str_list: list[str]) -> list[str]:
        str_list = list(filter(lambda e: len(e) > 0, str_list))
        return sorted(list(set(str_list)))

    test_dataset = MyDataset(test_data_path)
    for data in tqdm(test_dataset):

        name = data['name']
        model_labeled_grammar = model_labeled_grammar_dict[name]

        model_labeled_grammar['productions'] = [
            e for e in model_labeled_grammar['productions'] if len(e) > 0
        ]
        model_labeled_grammar['constraints'] = [
            e for e in model_labeled_grammar['constraints'] if len(e) > 0
        ]

        if len(model_labeled_grammar['productions']) != 0:
            model_labeled_grammar_stringified = (
                MyDataset.stringify(model_labeled_grammar))
            model_labeled_grammar = (
                stringified_to_grammar(model_labeled_grammar_stringified))

        model_labeled_productions = model_labeled_grammar['productions']
        model_labeled_productions = normalize_list(model_labeled_productions)
        model_labeled_constraints = model_labeled_grammar['constraints']
        model_labeled_constraints = normalize_list(model_labeled_constraints)

        human_labeled_grammar = stringified_to_grammar(data['stringified'])
        human_labeled_productions = human_labeled_grammar["productions"]
        human_labeled_productions = normalize_list(human_labeled_productions)
        human_labeled_constraints = human_labeled_grammar["constraints"]
        human_labeled_constraints = normalize_list(human_labeled_constraints)

        exact_match_productions.append(
            model_labeled_productions == human_labeled_productions)
        exact_match_constraints.append(
            model_labeled_constraints == human_labeled_constraints)

        logging.debug(f"Name: {name}")

        logging.debug("Specification:")
        logging.debug(data['specification'])

        logging.debug("Human Labeled Productions:")
        logging.debug(human_labeled_productions)
        logging.debug("Model Labeled Productions:")
        logging.debug(model_labeled_productions)

        logging.debug("Human Labeled Constraints:")
        logging.debug(human_labeled_constraints)
        logging.debug("Model Labeled Constraints:")
        logging.debug(model_labeled_constraints)

    exact_match = list(map(
        lambda e: e[0] and e[1],
        zip(exact_match_productions, exact_match_constraints)
    ))
    average_exact_match = sum(exact_match) / len(exact_match)
    average_exact_match_productions = (
        sum(exact_match_productions) / len(exact_match_productions))
    average_exact_match_constraints = (
        sum(exact_match_constraints) / len(exact_match_constraints))

    print(model_labeled_data_path)
    print('& Productions & Constraints & Grammar \\\\')
    print('& {:.2f} & {:.2f} & {:.2f} \\\\'.format(
        average_exact_match_productions * 100,
        average_exact_match_constraints * 100,
        average_exact_match * 100,
    ))


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-labeled-data')
    args = parser.parse_args()

    with open('./config.json') as fp:
        config = json.load(fp)

    defaults = {
        'model_labeled_data': args.model_labeled_data,
    }

    task = 'test'
    task_config = config.setdefault(task, {})
    for k in defaults.keys():
        if getattr(args, k, None) is not None:
            task_config[k] = getattr(args, k)
        task_config.setdefault(k, defaults[k])

    main(config)
