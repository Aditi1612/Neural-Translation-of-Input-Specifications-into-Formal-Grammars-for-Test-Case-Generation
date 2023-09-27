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

from data_loader import get_my_data_loader
from data_loader import MyDataset
from model import MyModel
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer
from trainer import MyModelTrainer
from pseudo_labeler import get_pseudo_labeler_correct
from pseudo_labeler import get_pseudo_labeler_sound
from pseudo_labeler import get_pseudo_labeler_complete
from pseudo_labeler import get_pseudo_labeler_base
from validator import get_soundness
from validator import get_completeness


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

    solution_prefix = Path(config['solution_prefix'])

    validate_labeling_config = config['validate_labeling']
    get_soundness_args = validate_labeling_config['get_soundness']['args']
    get_completeness_args = (
        validate_labeling_config['get_completeness']['args'])

    data_loader_args = config['data_loader']['args']
    data_dir = Path(config['data_dir'])
    solution_prefix = Path(config['solution_prefix'])
    pretrained = config['pretrained']
    trainer_args = config['trainer']
    optimizer_args = config['optimizer']['args']
    source_encoding_args = config['source_encoding']['args']

    # Set variables related to `train_config`
    train_config = config['train']
    logging.info(train_config)
    loss_path = Path(train_config['loss_path'])
    generation_config = GenerationConfig(**train_config['generation_config'])
    pseudo_labeler_config = train_config.get('pseudo_labeler', None)

    source_tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])
    target_tokenizer = CcfgTokenizer(source_tokenizer)

    train_data_path = data_dir / config['train_data']
    valid_data_path = data_dir / config['unlabeled_valid_data']
    unlabeled_train_data_path = data_dir / config['unlabeled_train_data']

    def get_testcases_dictionary(testcases_path: Path) -> dict[str, list[str]]:
        testcases_dictionary: dict[str, list[str]] = {}
        with jsonlines.open(testcases_path, 'r') as dataset:
            for data in tqdm(dataset, desc='Loading testcases'):
                name = data['name']
                testcases = data['public_tests']['input']
                testcases_dictionary[name] = testcases
        return testcases_dictionary

    valid_testcases_path = data_dir / config['unlabeled_valid_data']
    valid_testcases_dictionary = get_testcases_dictionary(valid_testcases_path)

    train_testcases_path = data_dir / config['unlabeled_train_data']
    train_testcases_dictionary = get_testcases_dictionary(train_testcases_path)

    train_data_loader = get_my_data_loader(
        train_data_path,
        source_tokenizer,
        target_tokenizer,
        source_encoding_args,
        **data_loader_args
    )

    def validate(model: MyModel) -> float:
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

        soundness = []
        completeness = []
        valid_dataset: list[dict[str, Any]] = []
        with jsonlines.open(valid_data_path, 'r') as f:
            valid_dataset.extend(f)

        for i, labeled_valid_data in enumerate(map(label, valid_dataset)):
            grammar = labeled_valid_data['grammar']
            name = labeled_valid_data['name']
            testcases = valid_testcases_dictionary[name]
            solution_dir = solution_prefix / name

            is_sound = get_soundness(
                grammar, solution_dir, name=name, **get_soundness_args)
            is_complete = get_completeness(
                grammar, testcases, name=name, **get_completeness_args)

            if i % np.ceil(len(valid_dataset) / 10) == 0:
                logging.debug(
                    "Validation {:.2f}%"
                    .format(i/len(valid_dataset) * 100)
                )

            soundness.append(is_sound)
            completeness.append(is_complete)
        correctness = list(map(lambda e: all(e), zip(completeness, soundness)))

        average_soundness = np.mean(soundness)
        average_completeness = np.mean(completeness)
        average_correctness = np.mean(correctness)

        print(f"Sound: {average_soundness * 100:.2f}%")
        print(f"Complete: {average_completeness * 100:.2f}%")
        print(f"Sound and Complete: {average_correctness * 100:.2f}%")

        return average_correctness

    unlabeled_data_list: list[dict[str, Any]] = []
    with jsonlines.open(unlabeled_train_data_path, 'r') as reader:
        unlabeled_data_list.extend(
            tqdm(reader, desc='Loading unlabeled train data'))

    production_model = T5ForConditionalGeneration.from_pretrained(pretrained)
    constraint_model = T5ForConditionalGeneration.from_pretrained(pretrained)
    model = MyModel(
        production_model, constraint_model, source_tokenizer, target_tokenizer)
    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), **optimizer_args)

    if pseudo_labeler_config is not None:
        pseudo_labeler_type = pseudo_labeler_config.get('type', 'correct')
        pseudo_labeler_args = pseudo_labeler_config.get('args', {})

        def get_testcases(name: str):
            return train_testcases_dictionary.get(name, [])

        get_solution_dir = solution_prefix.joinpath

        if pseudo_labeler_type == 'base':
            pseudo_labeler = get_pseudo_labeler_base(
                source_tokenizer,
                generation_config,
                device,
                source_encoding_args
            )
        elif pseudo_labeler_type == 'sound':
            pseudo_labeler = get_pseudo_labeler_sound(
                source_tokenizer,
                generation_config,
                device,
                source_encoding_args,
                get_solution_dir,
                **pseudo_labeler_args
            )
        elif pseudo_labeler_type == 'complete':
            pseudo_labeler = get_pseudo_labeler_complete(
                source_tokenizer,
                generation_config,
                device,
                source_encoding_args,
                get_testcases,
                **pseudo_labeler_args
            )
        elif pseudo_labeler_type == 'correct':
            pseudo_labeler = get_pseudo_labeler_correct(
                source_tokenizer,
                generation_config,
                device,
                source_encoding_args,
                get_solution_dir,
                get_testcases,
                **pseudo_labeler_args
            )
        else:
            raise ValueError(
                f"Unknown pseudo-labeler type: {pseudo_labeler_type}")
    else:
        pseudo_labeler = None

    trainer = MyModelTrainer(
        model,
        optimizer,
        device,
        train_data_loader,
        validate,
        unlabeled_data_list,
        pseudo_labeler=pseudo_labeler,
        **trainer_args)

    losses = trainer.train()
    with open(loss_path, 'w') as fp:
        json.dump(losses, fp)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    ccfg_logger = logging.getLogger('counting_context_free_grammar')
    ccfg_logger.setLevel(logging.INFO)

    validator_logger = logging.getLogger('validator')
    validator_logger.setLevel(logging.ERROR)

    data_loader_logger = logging.getLogger('data_loader.my_data_loader')
    data_loader_logger.setLevel(logging.WARNING)

    labeler_logger = logging.getLogger('pseudo_labeler')
    labeler_logger.setLevel(logging.WARNING)

    trainer_logger = logging.getLogger('trainer.my_model_trainer')
    trainer_logger.addHandler(logging.FileHandler('train.log'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--loss-path')
    parser.add_argument('--config', default='./config.json')
    args = parser.parse_args()

    with open(args.config) as fp:
        config = json.load(fp)

    defaults = {'loss_path': '/dev/null'}

    task = 'train'
    task_config = config.setdefault(task, {})
    for k in defaults.keys():
        if getattr(args, k, None) is not None:
            task_config[k] = getattr(args, k)
        task_config.setdefault(k, defaults[k])

    main(config)
