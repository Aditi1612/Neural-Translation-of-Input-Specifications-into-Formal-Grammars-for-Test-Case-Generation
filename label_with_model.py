import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import (Any, )

import jsonlines
import torch
import numpy as np
from tqdm import tqdm
from transformers import GenerationConfig
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


def main(config: dict[str, Any]) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Use device: {device}")

    pretrained_model_name = config['pretrained']
    source_encoding_args = config['source_encoding']['args']

    # Set variables related to `label_config`
    label_config = config['label']
    logging.info(label_config)
    if label_config['model_pth'] is None:
        checkpoint_path = None
        model_dir = Path(label_config['model_dir'])
        checkpoint_paths = model_dir.glob('*.pth')
        checkpoint_path = max(checkpoint_paths, key=os.path.getctime)
    else:
        checkpoint_path = Path(label_config['model_pth'])

    generation_config = GenerationConfig(**label_config['generation_config'])
    unlabeled_data_path = Path(label_config['unlabeled_data'])
    output_path = label_config['output']

    logging.info(f"Use device: {device}")
    logging.info(f"Dataset: {unlabeled_data_path}")
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

    def label(unlabeled: dict[str, Any]) -> dict[str, Any]:
        prefix = "summarize: "
        name = unlabeled['name']
        description = unlabeled['description']

        # Tokenize description
        specification = MyDataset.get_specification(description)
        encoding = source_tokenizer.encode(
            prefix + specification, **source_encoding_args)
        input_ids = encoding.to(device)

        # Generate grammar
        generated_productions_list, generated_constraints_list = (
            model.generate(input_ids, generation_config))
        productions = generated_productions_list[0]
        constraints = generated_constraints_list[0]
        grammar = {'productions': productions, 'constraints': constraints}

        labeled_data: dict[str, Any] = {}
        labeled_data['name'] = name
        labeled_data['description'] = description
        labeled_data['grammar'] = grammar

        return labeled_data

    unlabeled_dataset = jsonlines.open(unlabeled_data_path, 'r')
    labeled_dataset = map(label, unlabeled_dataset)
    assert os.path.exists(output_path) is False, "Output file already exists"
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(tqdm(labeled_dataset, desc='Labeling'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir')
    parser.add_argument('--model-pth')
    parser.add_argument('--unlabeled-data')
    parser.add_argument('--output')
    parser.add_argument('--config', default='./config.json')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    with open(args.config) as fp:
        config = json.load(fp)

    data_dir = Path(config['data_dir'])
    unlabeled_valid_data = data_dir / config['unlabeled_valid_data']
    unlabeled_test_data = data_dir / config['unlabeled_test_data']
    unlabeled_data = unlabeled_test_data if args.test else unlabeled_valid_data
    output_base = (
        'model_labeled_test_data.jsonl' if args.test
        else 'model_labeled_valid_data.jsonl'
    )

    defaults = {
        'model_dir': None,
        'model_pth': None,
        'unlabeled_data': unlabeled_data,
        'output': None
    }

    task = 'label'
    task_config = config.setdefault(task, {})
    for k in defaults.keys():
        if getattr(args, k, None) is not None:
            task_config[k] = getattr(args, k)
        task_config.setdefault(k, defaults[k])

    main(config)
