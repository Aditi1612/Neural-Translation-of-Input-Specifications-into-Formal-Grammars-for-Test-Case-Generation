import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import (Any, )

import torch
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import T5ForConditionalGeneration  # type: ignore [import]
from transformers import GenerationConfig

from data_loader import MyDataset
from model import MyModel
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer


# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)  # numpy random seed
random.seed(SEED)  # python random seed


def main(config: dict[str, Any]) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Use device: {device}")

    data_dir = Path(config['data_dir'])
    test_data_path = data_dir / config['test_data']
    pretrained_model_name = config['pretrained']

    # Set variables related to `test_config`
    test_config = config['test']
    logging.info(test_config)
    model_dir = Path(test_config['model_dir'])
    if test_config['model_pth'] is None:
        checkpoint_paths = model_dir.glob('*')
        checkpoint_path = max(checkpoint_paths, key=os.path.getctime)
    else:
        checkpoint_path = Path(test_config['model_pth'])
    generation_config = GenerationConfig(**test_config['generation_config'])

    logging.info(f"Use device: {device}")
    logging.info(f"Dataset: {test_data_path}")
    logging.info(f"Checkpoint: {checkpoint_path}")

    # Create a data loader
    source_tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
    target_tokenizer = CcfgTokenizer(source_tokenizer)

    # Load the model
    production_model = (
        T5ForConditionalGeneration
        .from_pretrained(pretrained_model_name)
    )
    constraint_model = (
        T5ForConditionalGeneration
        .from_pretrained(pretrained_model_name)
    )
    model = MyModel(
        production_model,
        constraint_model,
        source_tokenizer,
        target_tokenizer
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)

    source_encoding_args = config['source_encoding']['args']

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

    exact_match_productions: list[int] = []
    exact_match_constraints: list[int] = []

    def normalize_list(str_list: list[str]) -> list[str]:
        str_list = list(filter(lambda e: len(e) > 0, str_list))
        # str_list = list(map(str.lower, str_list))
        return sorted(list(set(str_list)))

    def log_result(
        exact_match: list[int],
        legend: str,
        *,
        k: int = 1,
        level=logging.INFO
    ) -> None:
        top_k_exact_match = [(k > e and e >= 0) for e in exact_match]
        top_k_sum_exact_match = sum(top_k_exact_match)
        top_k_len_exact_match = len(top_k_exact_match)
        average_exact_match = top_k_sum_exact_match / top_k_len_exact_match
        # logging.log(
        #     logging.INFO,
        #     "Top-{} exact match of {}: {:.3f}%({}/{})"
        #     .format(
        #         k, legend,
        #         average_exact_match*100,
        #         top_k_sum_exact_match,
        #         top_k_len_exact_match
        #     )
        # )
        print(f"{average_exact_match * 100:.2f}", end='')

    def find(item: Any, list_: list[Any]) -> int:
        try:
            return list_.index(item)
        except ValueError:
            return -1

    test_dataset = MyDataset(test_data_path)
    PREFIX = "summarize: "
    with torch.no_grad():
        for data in tqdm(test_dataset):

            name = data['name']
            specification = data['specification']
            encoding = source_tokenizer.encode(
                PREFIX + specification, **source_encoding_args)
            input_ids = encoding.to(device)

            generated_productions_list, generated_constraints_list = (
                model.generate(input_ids, generation_config))
            generated_productions_list = list(
                map(normalize_list, generated_productions_list))
            generated_constraints_list = list(
                map(normalize_list, generated_constraints_list))

            labeled_grammar = stringified_to_grammar(data['stringified'])
            labeled_productions = labeled_grammar["productions"]
            labeled_productions = normalize_list(labeled_productions)
            labeled_constraints = labeled_grammar["constraints"]
            labeled_constraints = normalize_list(labeled_constraints)

            exact_match_productions.append(
                find(labeled_productions, generated_productions_list))
            exact_match_constraints.append(
                find(labeled_constraints, generated_constraints_list))

            logging.debug(f"Name: {name}")

            logging.debug("Specification:")
            logging.debug(specification)

            logging.debug("Labeled Productions:")
            logging.debug(labeled_productions)
            logging.debug("Generated Productions:")
            logging.debug(generated_productions_list[0])

            logging.debug("Labeled Constraints:")
            logging.debug(labeled_constraints)
            logging.debug("Generated Constraints:")
            logging.debug(generated_constraints_list[0])

    exact_match = list(map(
        lambda e: max(e[0], e[1]) if (e[0] >= 0 and e[1] >= 0) else -1,
        zip(exact_match_productions, exact_match_constraints)
    ))

    print('& Top-1 & ', end='')
    log_result(exact_match, "grammar", k=1)
    print(' & ', end='')
    log_result(exact_match_productions, "productions", k=1)
    print(' & ', end='')
    log_result(exact_match_constraints, "constraints", k=1)
    print(' \\\\')

    print('& Top-5 & ', end='')
    log_result(exact_match, "grammar", k=5)
    print(' & ', end='')
    log_result(exact_match_productions, "productions", k=5)
    print(' & ', end='')
    log_result(exact_match_constraints, "constraints", k=5)
    print(' \\\\')

    print('& Top-10 & ', end='')
    log_result(exact_match, "grammar", k=10)
    print(' & ', end='')
    log_result(exact_match_productions, "productions", k=10)
    print(' & ', end='')
    log_result(exact_match_constraints, "constraints", k=10)
    print(' \\\\')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir')
    parser.add_argument('--model-pth')
    args = parser.parse_args()

    with open('./config.json') as fp:
        config = json.load(fp)

    trainer_config = config['trainer']
    model_dir = trainer_config['save_dir']

    defaults = {
        'model_dir': model_dir,
        'model_pth': None
    }

    task = 'test'
    task_config = config.setdefault(task, {})
    for k in defaults.keys():
        if getattr(args, k, None) is not None:
            task_config[k] = getattr(args, k)
        task_config.setdefault(k, defaults[k])

    main(config)
