import logging
from pathlib import Path
from typing import (Any, Optional, Callable, )

import torch
from transformers import GenerationConfig

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg
from data_loader import MyDataset
from grammar_tester import test_soundness
from grammar_tester import test_completeness
from grammar_tester import test_correctness
from model import MyModel
from tokenizer import Tokenizer
from trainer import PseudoLabeler


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
    num_testcase_generation=10
) -> PseudoLabeler:

    pseudo_labeler_base = get_pseudo_labeler_base(
        tokenizer, generation_config, device, encoding_args)

    def pseudo_labeler_generatable(
        self, unlabeled_data: dict[str, Any], model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        grammar = pseudo_labeler_base(unlabeled_data, model)
        try:
            for _ in range(num_testcase_generation):
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
    get_solution_dir: Callable[[str], Path],
    *,
    num_testcase_generation: Optional[int] = None,
) -> PseudoLabeler:

    pseudo_labeler_base = get_pseudo_labeler_base(
        tokenizer, generation_config, device, encoding_args)

    def pseudo_labeler_sound(
        unlabeled_data: dict[str, Any], model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        name = unlabeled_data['name']
        grammar = pseudo_labeler_base(unlabeled_data, model)
        is_sound = test_soundness(
            grammar, get_solution_dir(name),
            name=name,
            num_testcase_generation=num_testcase_generation,
        )
        if not is_sound:
            logging.debug("Pseudo labeling failed: Unsound grammar.")
            return None
        return grammar

    return pseudo_labeler_sound


def get_pseudo_labeler_complete(
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
    testcases_dictionary: dict[list[str]],
    *,
    num_testcase_sampling: Optional[int] = None,
) -> PseudoLabeler:

    pseudo_labeler_base = get_pseudo_labeler_base(
        tokenizer, generation_config, device, encoding_args)

    def pseudo_labeler_complete(
        unlabeled_data: dict[str, Any], model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        name = unlabeled_data['name']
        grammar = pseudo_labeler_base(unlabeled_data, model)
        testcases = testcases_dictionary[name]
        is_complete = test_completeness(
            grammar, testcases, name=name,
            num_testcase_sampling=num_testcase_sampling
        )
        if not is_complete:
            logging.debug("Pseudo labeling failed: Incomplete grammar.")
            return None
        return grammar

    return pseudo_labeler_complete


def get_pseudo_labeler_correct(
    tokenizer: Tokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
    get_solution_dir: Callable[[str], Path],
    get_testcases: Callable[[str], list[str]],
    *,
    num_testcase_generation: Optional[int],
    num_solution_sampling: Optional[int] = None,
    num_testcase_sampling: Optional[int] = None,
    timeout: float = 2,
) -> PseudoLabeler:

    pseudo_labeler_base = get_pseudo_labeler_base(
        tokenizer, generation_config, device, encoding_args)

    def pseudo_labeler_correct(
        unlabeled_data: dict[str, Any], model: MyModel
    ) -> Optional[dict[str, list[str]]]:
        name = unlabeled_data['name']
        grammar = pseudo_labeler_base(unlabeled_data, model)
        solution_dir = get_solution_dir(name)
        testcases = get_testcases(name)
        is_correct = test_correctness(
            grammar, solution_dir, testcases, name,
            num_testcase_generation=num_testcase_generation,
            num_solution_sampling=num_solution_sampling,
            num_testcase_sampling=num_testcase_sampling,
        )
        if not is_correct:
            logging.debug("Pseudo labeling failed: Incorrect grammar.")
            return None
        return grammar

    return pseudo_labeler_correct
