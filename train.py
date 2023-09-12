import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import (Any, )

import torch
import jsonlines
import numpy as np
from tqdm import tqdm
from transformers import GenerationConfig  # type: ignore [import]
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import T5ForConditionalGeneration  # type: ignore [import]

from data_loader import get_my_data_loader
from model import MyModel
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer
from trainer import MyModelTrainer
from pseudo_labeler import get_pseudo_labeler_correct


# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)  # numpy random seed


def main(config: dict[str, Any]) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Use device: {device}")

    data_loader_args = config['data_loader']['args']
    data_dir = Path(config['data_dir'])
    solution_prefix = Path(config['solution_prefix'])
    pretrained = config['pretrained']
    trainer_args = config['trainer']
    optimizer_args = config['optimizer']['args']
    train_config = config['train']
    generation_config = GenerationConfig(**train_config['generation_config'])
    loss_path = Path(train_config.get('loss_path', './dev/null'))
    source_encoding_args = config['source_encoding']['args']

    source_tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])
    target_tokenizer = CcfgTokenizer(source_tokenizer)

    train_data_path = data_dir / config['train_data']
    valid_data_path = data_dir / config['valid_data']
    unlabeled_data_path = data_dir / config['unlabeled_train_data']
    testcases_path = (
        data_dir / 'unlabeled' / 'code_contests_train_python.jsonl')

    testcases_dictionary: dict[str, list[str]] = {}
    with jsonlines.open(testcases_path, 'r') as dataset:
        for data in tqdm(dataset, desc='Loading testcases'):
            name = data['name']
            testcases = data['public_tests']['input']
            # testcases.extend(data['private_tests']['input'])
            # testcases.extend(data['generated_tests']['input'])
            testcases_dictionary[name] = testcases

    train_data_loader = get_my_data_loader(
        train_data_path,
        source_tokenizer,
        target_tokenizer,
        source_encoding_args,
        **data_loader_args
    )

    valid_data_loader = get_my_data_loader(
        valid_data_path,
        source_tokenizer,
        target_tokenizer,
        source_encoding_args,
        **data_loader_args
    )

    unlabeled_data_list: list[dict[str, Any]] = []
    with jsonlines.open(unlabeled_data_path, 'r') as reader:
        unlabeled_data_list.extend(tqdm(reader, desc='Loading unlabeled data'))

    production_model = T5ForConditionalGeneration.from_pretrained(pretrained)
    constraint_model = T5ForConditionalGeneration.from_pretrained(pretrained)
    model = MyModel(
        production_model, constraint_model, source_tokenizer, target_tokenizer)
    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), **optimizer_args)
    pseudo_labeler = get_pseudo_labeler_correct(
        source_tokenizer,
        generation_config,
        device,
        source_encoding_args,
        get_solution_dir=solution_prefix.joinpath,
        get_testcases=lambda name: testcases_dictionary.get(name, []),
        num_testcase_generation=10,
        num_solution_sampling=10,
        num_testcase_sampling=10
    )

    trainer = MyModelTrainer(
        model,
        optimizer,
        device,
        train_data_loader,
        valid_data_loader,
        unlabeled_data_list,
        pseudo_labeler=pseudo_labeler,
        **trainer_args)

    losses = trainer.train()
    with open(loss_path, 'w') as fp:
        json.dump(losses, fp)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    ccfg_logger = logging.getLogger('counting_context_free_grammar')
    ccfg_logger.setLevel(logging.INFO)

    with open('./config.json') as fp:
        config = json.load(fp, object_hook=OrderedDict)
    main(config)
