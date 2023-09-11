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
from data_loader import MyDataset
from grammar_tester import test_soundness
from model import MyModel
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer
from tokenizer import Tokenizer
from trainer import MyModelTrainer
from trainer import PseudoLabeler


# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)  # numpy random seed


def get_pseudo_labeler_base(
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
) -> PseudoLabeler:

    PREFIX = "summarize: "

    def pseudo_labeler_base(
        unlabeled_data: dict[str, Any], model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        description = unlabeled_data['description']
        specification = MyDataset.get_specification(description)
        encoding = tokenizer.encode(PREFIX + specification, **encoding_args)
        input_ids = encoding.to(device)

        grammar = model.generate(input_ids, generation_config)
        logging.debug("Grammar:")
        logging.debug(grammar)
        return grammar

    return pseudo_labeler_base


def get_pseudo_labeler_compilable(
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
) -> PseudoLabeler:

    pseudo_labeler_base = get_pseudo_labeler_base(
        tokenizer, generation_config, device, encoding_args)

    def pseudo_labeler_compilable(
        unlabeled_data: dict[str, Any], model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        grammar = pseudo_labeler_base(unlabeled_data, model)
        try:
            Ccfg(**grammar)
        except Exception as e:
            logging.debug(f"Pseudo labeling failed: {e}")
            return None
        return grammar

    return pseudo_labeler_compilable


def get_pseudo_labeler_generatable(
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
    *,
    num_testcases=10
) -> PseudoLabeler:

    pseudo_labeler_base = get_pseudo_labeler_base(
        tokenizer, generation_config, device, encoding_args)

    def pseudo_labeler_generatable(
        self, unlabeled_data: dict[str, Any], model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        grammar = pseudo_labeler_base(unlabeled_data, model)
        try:
            for _ in range(num_testcases):
                Ccfg(**grammar, testmode=True).generate()
        except Exception as e:
            logging.debug(f"Pseudo labeling failed: {e}")
            return None
        return grammar

    return pseudo_labeler_generatable


def get_pseudo_labeler_sound(
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
    solution_dir: Path,
    *,
    num_testcases: Optional[int] = None,
) -> PseudoLabeler:

    pseudo_labeler_base = get_pseudo_labeler_base(
        tokenizer, generation_config, device, encoding_args)

    def pseudo_labeler_sound(
        unlabeled_data: dict[str, Any], model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        name = unlabeled_data['name']
        grammar = pseudo_labeler_base(unlabeled_data, model)
        is_sound = test_soundness(
            grammar, solution_dir, name, num_testcases=num_testcases)
        if not is_sound:
            logging.debug("Pseudo labeling failed: Unsound grammar.")
            return None
        return grammar

    return pseudo_labeler_sound


def main(config: dict[str, Any]) -> None:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Use device: {device}")

    source_tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])
    target_tokenizer = CcfgTokenizer(source_tokenizer)

    data_loader_args = config['data_loader']['args']
    data_dir = Path(config['data_dir'])
    solution_dir = Path(config['solution_dir'])
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
    pseudo_labeler = get_pseudo_labeler_sound(
        source_tokenizer,
        generation_config,
        device,
        source_encoding_args,
        solution_dir,
        num_testcases=10
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
    trainer.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    ccfg_logger = logging.getLogger('counting_context_free_grammar')
    ccfg_logger.setLevel(logging.INFO)

    with open('./config.json') as fp:
        config = json.load(fp, object_hook=OrderedDict)
    main(config)
