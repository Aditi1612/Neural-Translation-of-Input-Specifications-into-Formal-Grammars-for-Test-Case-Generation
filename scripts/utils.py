"""This module contains utility functions."""

import itertools
import os
from pathlib import Path
import random
from typing import Iterable, Optional, TypedDict, TypeVar, Any

import jsonlines
from transformers import RobertaTokenizer  # type: ignore

from data_loader import MyDataset  # type: ignore
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer  # type: ignore


_model_name = "Salesforce/codet5-base"
_source_tokenizer = RobertaTokenizer.from_pretrained(_model_name)
_target_tokenizer = CcfgTokenizer(_source_tokenizer)

T = TypeVar("T")


def stringified_to_grammar(
    stringified: str, target_tokenizer: CcfgTokenizer
) -> dict[str, list[str]]:
    production_encoding, constraint_encoding = (
        target_tokenizer.encode_to_splited(stringified)
    )

    subseparator = target_tokenizer.subseparator
    production_decoding = target_tokenizer.decode(production_encoding)
    constraint_decoding = target_tokenizer.decode(constraint_encoding)

    productions = production_decoding.split(subseparator)
    productions = list(map(str.strip, productions))
    constraints = constraint_decoding.split(subseparator)
    constraints = list(map(str.strip, constraints))

    grammar = {"productions": productions, "constraints": constraints}
    return grammar


def normalize_grammar(grammar: dict[str, list[str]]) -> dict[str, list[str]]:
    stringified = MyDataset.stringify(grammar)
    grammar = stringified_to_grammar(stringified, _target_tokenizer)
    grammar["constraints"] = normalize_constraints(grammar["constraints"])
    return grammar


def normalize_productions(productions: list[str]) -> list[str]:
    stringified = MyDataset.stringify(
        {"productions": productions, "constraints": []}
    )
    grammar = stringified_to_grammar(stringified, _target_tokenizer)
    return grammar["productions"]


def normalize_constraints(constraints: list[str]) -> list[str]:
    constraints = [e for e in constraints if len(e) > 0]
    if len(constraints) == 0:
        return []
    return sorted(["".join(e.split()) for e in constraints])


def get_mode(xs: list[str]) -> tuple[str, int]:
    groupby_iterable = itertools.groupby(sorted(xs))
    groups = [(k, len(list(v))) for k, v in groupby_iterable]
    groups = sorted(groups, key=lambda e: e[1], reverse=True)
    mode, num_of_mode = groups[0]

    return mode, num_of_mode


def sanitize(xs: Iterable[T], filename: str = "test") -> Iterable[T]:
    ground_truth_generation_results = jsonlines.open(
        Path(os.environ["GROUND_TRUTH_GENERATION_RESULT"]), "r"
    )
    public_generation_results = jsonlines.open(
        Path(os.environ["PUBLIC_GENERATION_RESULT"]), "r"
    )
    private_generation_results = jsonlines.open(
        Path(os.environ["PRIVATE_GENERATION_RESULT"]), "r"
    )
    ground_truth_parsing_results = jsonlines.open(
        Path(os.environ["GROUND_TRUTH_PARSING_RESULT"]), "r"
    )
    ground_truth_execution_summaries = jsonlines.open(
        Path(os.environ["GROUND_TRUTH_EXECUTION_SUMMARY"]), "r"
    )

    it = zip(
        xs,
        ground_truth_generation_results,
        public_generation_results,
        private_generation_results,
        ground_truth_parsing_results,
        ground_truth_execution_summaries,
    )
    for (
        x,
        ground_truth_generation_result,
        public_generation_result,
        private_generation_result,
        ground_truth_parsing_result,
        ground_truth_execution_summary,
    ) in it:
        name = ground_truth_generation_result["name"]
        assert name == public_generation_result["name"]
        assert name == private_generation_result["name"]
        assert name == ground_truth_parsing_result["name"]
        assert name == ground_truth_execution_summary["name"]

        # Exclude problems without testcase of ground truth grammar
        if len(ground_truth_generation_result["results"]) == 0:
            continue

        # Exclude problems with wrong generation result
        flag = False
        for result in [
            ground_truth_generation_result["results"],
            public_generation_result["results"],
            private_generation_result["results"],
            ground_truth_parsing_result["results"],
        ]:
            if not all(e["parsable"] for e in result):
                flag = True
                break
        if flag:
            continue

        # Exclude problems without incorrect solutions
        summary_results = ground_truth_execution_summary["results"]
        assert len(summary_results) > 0
        if len(summary_results[0]["incorrect_results"]) == 0:
            continue

        yield x


def sample_solutions(
    name: str, correctness: str, k: int, seed: int
) -> list[Path]:
    if correctness == "correct":
        solution_dir = Path(os.environ["CORRECT_SOLUTIONS_DIR"]) / name
    elif correctness == "incorrect":
        solution_dir = Path(os.environ["INCORRECT_SOLUTIONS_DIR"]) / name
    else:
        raise ValueError(f"Invalid correctness: {correctness}")

    random.seed(seed)
    solutions = list(solution_dir.glob("*.py"))
    k = min(k, len(solutions))
    return random.sample(solutions, k)


def get_timeout_dict(filename: str) -> dict[str, int]:
    public_dataset = jsonlines.open(
        Path(os.environ["PUBLIC_TESTCASE_DIR"]) / f"{filename}.jsonl", "r"
    )
    timeout_dict = {
        obj["name"]: max(
            1,
            int(
                obj["time_limit"]["seconds"] + obj["time_limit"]["nanos"] / 1e9
            ),
        )
        for obj in public_dataset
    }
    return timeout_dict


def get_testcase_num_dict(filename: str) -> dict[str, int]:
    public_dataset_path = (
        Path(os.environ["PUBLIC_TESTCASE_DIR"]) / f"{filename}.jsonl"
    )
    private_dataset_path = (
        Path(os.environ["PRIVATE_TESTCASE_DIR"]) / f"{filename}.jsonl"
    )

    public_dataset = jsonlines.open(public_dataset_path, "r")
    private_dataset = jsonlines.open(private_dataset_path, "r")
    testcase_num_dict = {
        public_object["name"]: len(public_object["testcase"])
        + len(private_object["testcase"])
        for public_object, private_object in zip(
            public_dataset, private_dataset
        )
    }
    return testcase_num_dict


def split_with_level(xs: Iterable[T], filename: str) -> dict[int, list[T]]:
    levels: Iterable[dict[str, Any]] = jsonlines.open(
        Path(os.environ["LEVEL_DIR"]) / f"{filename}.jsonl"
    )
    levels = list(sanitize(levels))
    xs = list(xs)
    assert len(levels) == len(xs)

    level_to_xs: dict[int, list[T]] = {level: [] for level in range(1, 6)}
    for x, level_data in zip(xs, levels):
        level = level_data["max_level"]
        level_to_xs[level].append(x)

    return level_to_xs


class Grammar(TypedDict):
    productions: list[str]
    constraints: list[str]


class GrammarData(TypedDict):
    """GrammarData is a dictionary that contains the grammar information of a
    single problem.
    """

    name: str
    description: str
    grammar: Grammar


class GenerationResultPerTestcase(TypedDict):
    """GenerationResultPerTestcase is a dictionary that contains the results of
    generation for a single testcase.
    """

    parsable: bool
    error: Optional[str]


class GenerationResult(TypedDict):
    """GenerationResult is a dictionary that contains the results of generation
    for a single problem.
    """

    name: str
    results: list[GenerationResultPerTestcase]


ParsingResult = GenerationResult


class ExecutionResultPerTestcase(TypedDict):
    """ExecutionResultPerTestcase is a dictionary that contains the results of
    execution for a single testcase.
    """

    correct_outputs: list[Optional[str]]
    incorrect_outputs: list[Optional[str]]
    correct_solutions: list[str]
    incorrect_solutions: list[str]


class ExecutionResult(TypedDict):
    """ExecutionResult is a dictionary that contains the results of execution
    for a single problem.
    """

    name: str
    results: list[ExecutionResultPerTestcase]


class ExecutionSummaryPerTestcase(TypedDict):
    """ExecutionSummaryPerTestcase is a dictionary that contains the results of
    execution for a single testcase. It is used for summarizing the results of
    execution. If there is no correct_solution generating output, answer is None.
    """

    answer: Optional[str]
    correct_results: list[bool]
    incorrect_results: list[bool]


class ExecutionSummary(TypedDict):
    """ExecutionSummary is a dictionary that contains the results of execution
    for a single problem. It is used for summarizing the results of execution.
    """

    name: str
    results: list[ExecutionSummaryPerTestcase]


class CoverageEntry(TypedDict):
    """CoverageEntry is a dictionary that contains the coverage information for
    a single solution.
    """

    covered_lines: int
    num_statements: int


class CoverageResultPerTestcase(TypedDict):
    """CoverageResultPerTestcase is a dictionary that contains the coverage
    information for a single testcase.
    """

    correct_coverages: Optional[list[CoverageEntry]]
    incorrect_coverages: Optional[list[CoverageEntry]]
    correct_error: Optional[str]
    incorrect_error: Optional[str]
    correct_solutions: list[str]
    incorrect_solutions: list[str]


class CoverageResult(TypedDict):
    """CoverageResult is a dictionary that contains the coverage information for
    a single problem.
    """

    name: str
    results: list[CoverageResultPerTestcase]
