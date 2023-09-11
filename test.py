import argparse
import json
import logging
import os
from pathlib import Path
from collections import OrderedDict
from typing import (Any, )

import torch
from tqdm import tqdm
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import T5ForConditionalGeneration  # type: ignore [import]
from transformers import GenerationConfig

from data_loader import MyDataset
from model import MyModel
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer


def main(config: dict[str, Any]) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_config = config['test']
    data_dir = Path(config['data_dir'])
    pretrained_model_name = config['pretrained']
    model_dir = Path(
        test_config['model_dir']
        if test_config['model_dir'] is not None
        else config['trainer']['save_dir']
    )
    generation_config = GenerationConfig(**test_config['generation_config'])
    test_data_path = data_dir / config['test_data']

    checkpoint_paths = model_dir.glob('*')
    latest_checkpoint_path = max(checkpoint_paths, key=os.path.getctime)

    logging.info(f"Use device: {device}")
    logging.info(f"Dataset: {test_data_path}")
    logging.info(f"Checkpoint: {latest_checkpoint_path}")

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
    checkpoint = torch.load(latest_checkpoint_path, map_location=device)
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

    exact_match_productions = []
    exact_match_constraints = []

    def normalize_list(str_list: list[str]) -> list[str]:
        str_list = list(filter(lambda e: len(e) > 0, str_list))
        str_list = list(map(str.lower, str_list))
        return sorted(list(set(str_list)))

    def log_result(
        exact_match: list[bool],
        legend: str,
        level=logging.INFO
    ) -> None:
        sum_exact_match = sum(exact_match)
        len_exact_match = len(exact_match)
        average_exact_match = sum_exact_match / len_exact_match
        logging.log(
            logging.INFO,
            "Exact match of {}: {:.3f}%({}/{})"
            .format(
                legend,
                average_exact_match*100,
                sum_exact_match,
                len_exact_match
            )
        )

    test_dataset = MyDataset(test_data_path)
    PREFIX = "summarize: "
    with torch.no_grad():
        for num, data in enumerate(tqdm(test_dataset)):

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

            exact_match_constraints.append(any(map(
                lambda generated_constraints:
                    generated_constraints == labeled_constraints,
                generated_constraints_list
            )))
            exact_match_productions.append(any(map(
                lambda generated_productions:
                    generated_productions == labeled_productions,
                generated_productions_list
            )))

            logging.debug(f"Data {num} Specification:")
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
        lambda e: e[0] and e[1],
        zip(exact_match_productions, exact_match_constraints)
    ))
    log_result(exact_match_productions, "productions")
    log_result(exact_match_constraints, "constraints")
    log_result(exact_match, "grammar")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    with open('./config.json') as fp:
        config = json.load(fp, object_hook=OrderedDict)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    for k, v in vars(args).items():
        if v is not None:
            config['test'][k] = v
    logging.basicConfig(level=logging.INFO)
    main(config)
