import logging
import itertools
from pathlib import Path
from typing import (
    Any,
    Optional,
    Callable,
)

import torch
from transformers import GenerationConfig  # type: ignore [import]
from transformers import RobertaTokenizer  # type: ignore [import]

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg
from data_loader import MyDataset
from validator import get_soundness
from validator import get_completeness
from validator import get_correctness
from model import MyModel
from trainer import PseudoLabeler


Strategy = Callable[[dict[str, list[str]]], bool]
Data = dict[str, Any]
Grammar = dict[str, list[str]]
Generation = tuple[list[list[str]], list[list[str]]]

logger = logging.getLogger(__name__)


def _get_unlabeled_data_to_generation(
    tokenizer: RobertaTokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
) -> Callable[[Data, MyModel], Generation]:

    def _unlabeled_data_to_generation(
        unlabeled_data: Data, model: MyModel
    ) -> Generation:
        PREFIX = "summarize: "

        description = unlabeled_data["description"]
        specification = MyDataset.get_specification(description)
        input_ids = tokenizer.encode(PREFIX + specification, **encoding_args)
        input_ids = input_ids.to(device)
        generation = model.generate(input_ids, generation_config)

        productions = generation[0][0]
        constraints = generation[1][0]
        grammar = {"productions": productions, "constraints": constraints}

        logger.debug("Specification:")
        logger.debug(specification)
        logger.debug("Top-1 Grammar:")
        logger.debug(str(grammar))

        return generation

    return _unlabeled_data_to_generation


def _get_labeler(
    tokenizer: RobertaTokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
    strategy_builder: Callable[[Data], Strategy],
) -> PseudoLabeler:

    generate = _get_unlabeled_data_to_generation(
        tokenizer, generation_config, device, encoding_args
    )

    def labeler(unlabeled_data: Data, model: MyModel) -> Optional[Grammar]:

        generation = generate(unlabeled_data, model)
        strategy = strategy_builder(unlabeled_data)
        productions_list, constraints_list = generation

        for productions, constraints in itertools.product(
            productions_list, constraints_list
        ):
            grammar = {"productions": productions, "constraints": constraints}
            if strategy(grammar):
                return grammar
        return None

    return labeler


def get_pseudo_labeler_base(
    tokenizer: RobertaTokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
) -> PseudoLabeler:
    def strategy_builder(unlabeled_data: Data) -> Strategy:
        return lambda _: True

    labeler = _get_labeler(
        tokenizer, generation_config, device, encoding_args, strategy_builder
    )
    return labeler


def get_pseudo_labeler_compilable(
    tokenizer: RobertaTokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
) -> PseudoLabeler:

    def strategy_builder(unlabeled_data: Data) -> Strategy:
        def strategy(grammar: Grammar) -> bool:
            productions = grammar["productions"]
            constraints = grammar["constraints"]
            try:
                Ccfg(productions, constraints, testmode=True)
            except Exception:
                return False
            return True

        return strategy

    labeler = _get_labeler(
        tokenizer, generation_config, device, encoding_args, strategy_builder
    )
    return labeler


def get_pseudo_labeler_generatable(
    tokenizer: RobertaTokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
    *,
    num_testcase_generation=10,
) -> PseudoLabeler:

    def strategy_builder(unlabeled_data: Data) -> Strategy:
        def strategy(grammar: Grammar) -> bool:
            productions = grammar["productions"]
            constraints = grammar["constraints"]
            try:
                ccfg = Ccfg(productions, constraints, testmode=True)
                for _ in range(num_testcase_generation):
                    ccfg.generate()
            except Exception:
                return False
            return True

        return strategy

    labeler = _get_labeler(
        tokenizer, generation_config, device, encoding_args, strategy_builder
    )
    return labeler


def get_pseudo_labeler_sound(
    tokenizer: RobertaTokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
    get_solution_dir: Callable[[str], Path],
    *,
    num_testcase_generation: int,
) -> PseudoLabeler:

    def strategy_builder(unlabeled_data: Data) -> Strategy:
        name = unlabeled_data["name"]
        solution_dir = get_solution_dir(name)

        def strategy(grammar: Grammar) -> bool:
            return get_soundness(
                grammar,
                solution_dir,
                num_testcase_generation=num_testcase_generation,
            )

        return strategy

    labeler = _get_labeler(
        tokenizer, generation_config, device, encoding_args, strategy_builder
    )
    return labeler


def get_pseudo_labeler_complete(
    tokenizer: RobertaTokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
    get_testcases: Callable[[str], list[str]],
    *,
    num_testcase_sampling: Optional[int] = None,
) -> PseudoLabeler:

    def strategy_builder(unlabeled_data: Data) -> Strategy:
        name = unlabeled_data["name"]
        testcases = get_testcases(name)

        def strategy(grammar: Grammar) -> bool:
            is_complete = get_completeness(
                grammar,
                testcases,
                name=name,
                num_testcase_sampling=num_testcase_sampling,
            )
            return is_complete

        return strategy

    labeler = _get_labeler(
        tokenizer, generation_config, device, encoding_args, strategy_builder
    )
    return labeler


def get_pseudo_labeler_correct(
    tokenizer: RobertaTokenizer,
    generation_config: GenerationConfig,
    device: torch.device,
    encoding_args: dict[str, Any],
    get_solution_dir: Callable[[str], Path],
    get_testcases: Callable[[str], list[str]],
    *,
    num_testcase_generation: int,
    num_solution_sampling: Optional[int] = None,
    num_testcase_sampling: Optional[int] = None,
    timeout: float = 2,
) -> PseudoLabeler:

    def strategy_builder(unlabeled_data: dict[str, Any]) -> Strategy:
        name = unlabeled_data["name"]
        solution_dir = get_solution_dir(name)
        testcases = get_testcases(name)

        def strategy(grammar: Grammar) -> bool:
            is_correct = get_correctness(
                grammar,
                solution_dir,
                testcases,
                name,
                num_testcase_generation=num_testcase_generation,
                num_solution_sampling=num_solution_sampling,
                num_testcase_sampling=num_testcase_sampling,
            )
            return is_correct

        return strategy

    labeler = _get_labeler(
        tokenizer, generation_config, device, encoding_args, strategy_builder
    )
    return labeler
