import argparse
import json
import logging
import random
from pathlib import Path
from typing import (Any, )

import jsonlines
import torch
import numpy as np
from tqdm import tqdm
from transformers import GenerationConfig  # type: ignore [import]
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import T5ForConditionalGeneration  # type: ignore [import]

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


def main(
    checkpoint_path: Path,
    data_path: Path,
    output_path: Path,
    num_beams: int,
    config: dict[str, Any]
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pretrained_model_name = config['pretrained']
    source_encoding_args = config['source_encoding']['args']

    # Set variables related to `label_config`
    config['test']['generation_config']['num_beams'] = num_beams
    config['test']['generation_config']['num_return_sequences'] = num_beams
    generation_config = GenerationConfig(**config['test']['generation_config'])

    logging.info(f"Use device: {device}")
    logging.info(f"Dataset: {data_path}")
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

    def get_grammar_data(desc_dataset: dict[str, Any]) -> dict[str, Any]:
        prefix = "summarize: "
        name = desc_dataset['name']
        description = desc_dataset['description']

        # Tokenize description
        specification = MyDataset.get_specification(description)
        encoding = source_tokenizer.encode(
            prefix + specification, **source_encoding_args)
        input_ids = encoding.to(device)

        # Generate grammar
        generated_productions_list, generated_constraints_list = (
            model.generate(input_ids, generation_config))

        grammar_candidates = {
            'productions': generated_productions_list,
            'constraints': generated_constraints_list
        }

        grammar_data: dict[str, Any] = {}
        grammar_data['name'] = name
        grammar_data['description'] = description
        grammar_data['grammar_candidates'] = grammar_candidates
        return grammar_data

    desc_dataset = jsonlines.open(data_path, 'r')
    grammar_dataset = map(get_grammar_data, desc_dataset)
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(tqdm(grammar_dataset, desc='Grammar Generation'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-pth', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-beams', type=int)
    parser.add_argument('--config', default='./config.json')
    args = parser.parse_args()

    with open(args.config) as fp:
        config = json.load(fp)

    main(
        args.model_pth,
        args.data,
        args.output,
        args.num_beams,
        config
    )
