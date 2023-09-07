import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import (Any, Optional, )

import torch
import jsonlines
import numpy as np
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import T5ForConditionalGeneration  # type: ignore [import]
from transformers import GenerationConfig

from counting_context_free_grammar import CountingContextFreeGrammar as CCFG
from data_loader import get_my_data_loader
from model import MyModel
from tokenizer import CountingContextFreeGrammarTokenizer as CCFGTokenizer
from trainer import MyModelTrainer


# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)  # numpy random seed


def main() -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Use device: {device}")

    with open('./config.json') as fp:
        config = json.load(fp, object_hook=OrderedDict)

    source_tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])
    target_tokenizer = CCFGTokenizer(source_tokenizer)

    data_loader_args = config['data_loader']['args']
    data_dir = Path(config['data_dir'])
    train_data_path = data_dir / config['train_data']
    valid_data_path = data_dir / config['valid_data']
    unlabeled_data_path = data_dir / config['unlabeled_train_data']

    train_data_loader = get_my_data_loader(
        train_data_path, source_tokenizer, target_tokenizer,
        **data_loader_args
    )

    valid_data_loader = get_my_data_loader(
        valid_data_path, source_tokenizer, target_tokenizer,
        **data_loader_args
    )

    unlabeled_data_list: list[dict[str, Any]] = []
    with jsonlines.open(unlabeled_data_path, 'r') as reader:
        unlabeled_data_list.extend(reader)

    production_model = (
        T5ForConditionalGeneration.from_pretrained(config['pretrained']))
    constraint_model = (
        T5ForConditionalGeneration.from_pretrained(config['pretrained']))
    model = MyModel(
        production_model,
        constraint_model,
        source_tokenizer,
        target_tokenizer
    )
    model = model.to(device)

    optimizer_args = config['optimizer']['args']
    optimizer = torch.optim.Adam(params=model.parameters(), **optimizer_args)

    generation_config = GenerationConfig(**config['generation_config'])
    source_encoding_args = {
        'add_special_tokens': False,
        'max_length': 512,
        'padding': True,
        'return_tensors': 'pt',
        'truncation': True,
    }
    PREFIX = "summarize: "

    def pseudo_labeler(
        specification: str,
        model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        logging.info("Pseudo labeling...")

        encoding = source_tokenizer.encode(
            PREFIX + specification, **source_encoding_args)
        input_ids = encoding.to(device)

        grammar = model.generate(input_ids, generation_config)
        try:
            return CCFG(grammar)
        except Exception:
            return None

    trainer_args = config['trainer']
    trainer = MyModelTrainer(
        model,
        optimizer,
        device,
        train_data_loader,
        valid_data_loader,
        unlabeled_data_list,
        pseudo_labeler=pseudo_labeler,
        **trainer_args)
    trainer.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
