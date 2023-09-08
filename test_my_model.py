import argparse
import json
import logging
import os
from glob import glob
from pathlib import Path
from collections import OrderedDict
from typing import (Any, )

import torch
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import T5ForConditionalGeneration  # type: ignore [import]
from transformers import GenerationConfig

from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer
from counting_context_free_grammar import CountingContextFreeGrammar as CCFG
from model import MyModel
from data_loader import MyDataset


def main(config: dict[str, Any]) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_config = config['test']
    data_dir = Path(config['data_dir'])
    test_data_path = data_dir / config['test_data']
    pretrained_model_name = config['pretrained']

    if test_config['model_dir'] is None:
        model_dir = Path(config['trainer']['save_dir'])
    else:
        model_dir = Path(test_config['model_dir'])

    checkpoint_paths = glob(str(model_dir / '*'))
    latest_checkpoint_path = max(checkpoint_paths, key=os.path.getctime)

    generation_config = GenerationConfig(**config['generation_config'])

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

    def strinified_to_grammar(stringified: str):
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

    test_dataset = MyDataset(test_data_path)
    PREFIX = "summarize: "
    with torch.no_grad():
        for data in test_dataset:

            specification = data['specification']
            encoding = source_tokenizer.encode(
                PREFIX + specification, **source_encoding_args)
            input_ids = encoding.to(device)

            generated_grammar = model.generate(input_ids, generation_config)
            grammar = strinified_to_grammar(data['stringified'])

            print("Goal:")
            print(grammar)

            print("Output:")
            try:
                CCFG(**generated_grammar)
            except Exception as e:
                logging.warning("Invalid grammar")
                logging.warning(e)
            print(generated_grammar)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
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
