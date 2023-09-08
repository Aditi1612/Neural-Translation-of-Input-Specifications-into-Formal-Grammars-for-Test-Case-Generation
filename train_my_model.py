import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import (Any, Optional, )

import torch
import jsonlines
import numpy as np
from tqdm import tqdm
from transformers import GenerationConfig
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import T5ForConditionalGeneration  # type: ignore [import]

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg
from data_loader import get_my_data_loader
from model import MyModel
from tokenizer import Tokenizer
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer
from trainer import MyModelTrainer
from trainer import PseudoLabeler


# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)  # numpy random seed

PREFIX = "summarize: "


def get_pseudo_labeler_compilable(
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
) -> PseudoLabeler:

    def pseudo_labeler(
        specification: str,
        model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        encoding = tokenizer.encode(
            PREFIX + specification, **encoding_args)
        input_ids = encoding.to(device)

        grammar = model.generate(input_ids, generation_config)
        logging.debug("Grammar:")
        logging.debug(grammar)
        try:
            Ccfg(**grammar)
            return grammar
        except Exception as e:
            logging.debug("Pseudo labeling failed.")
            logging.debug(e)
            return None

    return pseudo_labeler


def get_pseudo_labeler_generatability(
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
) -> PseudoLabeler:

    pseudo_labeler_compilable = get_pseudo_labeler_compilable(
        tokenizer, generation_config, device, encoding_args)

    def pseudo_labeler_generatability(
        specification: str,
        model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        grammar = pseudo_labeler_compilable(specification, model)
        try:
            for _ in range(10):
                Ccfg(**grammar, testmode=True).generate()
            return grammar
        except Exception as e:
            logging.debug("Pseudo labeling failed.")
            logging.debug(e)
            return None

    return pseudo_labeler_generatability


def get_pseudo_labeler_soundness(
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
) -> PseudoLabeler:

    pseudo_labeler_compilable = get_pseudo_labeler_compilable(
        tokenizer, generation_config, device, encoding_args)

    def pseudo_labeler_soundness(
        specification: str,
        model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        grammar = pseudo_labeler_compilable(specification, model)
        try:
            raise NotImplementedError("TODO: Implement soundness check")
            return grammar
        except Exception as e:
            logging.debug("Pseudo labeling failed.")
            logging.debug(e)
            return None

    return pseudo_labeler_soundness


def main(config: dict[str, Any]) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Use device: {device}")

    source_tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])
    target_tokenizer = CcfgTokenizer(source_tokenizer)

    data_loader_args = config['data_loader']['args']
    data_dir = Path(config['data_dir'])
    pretrained = config['pretrained']
    trainer_args = config['trainer']
    optimizer_args = config['optimizer']['args']
    generation_config = GenerationConfig(**config['generation_config'])
    source_encoding_args = config['source_encoding']['args']

    train_data_path = data_dir / config['train_data']
    valid_data_path = data_dir / config['valid_data']
    unlabeled_data_path = data_dir / config['unlabeled_train_data']

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
    pseudo_labeler = get_pseudo_labeler_generatability(
        source_tokenizer, generation_config, device, source_encoding_args)

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
    with open('./config.json') as fp:
        config = json.load(fp, object_hook=OrderedDict)
    main(config)
