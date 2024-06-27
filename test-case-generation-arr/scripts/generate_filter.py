import argparse
import logging

import jsonlines
import timeout_decorator  # type: ignore
from tqdm import tqdm

from counting_context_free_grammar import Discriminator


@timeout_decorator.timeout(1)
def _check_syntactic_validness(
    testcase: str,
    grammar: dict[str, list[str]],
) -> bool:
    d = Discriminator()
    try:
        productions = grammar['productions']
        constraints = grammar['constraints']
        return d(productions, constraints, testcase)
    except Exception:
        return False


def check_syntactic_validness(
    testcase: str,
    description: str,
    grammar: dict[str, list[str]],
) -> bool:
    try:
        validness = _check_syntactic_validness(testcase, grammar)
        if not validness:
            logging.debug(description)
            logging.debug(testcase)
            logging.debug(str(grammar))
        return validness
    except timeout_decorator.TimeoutError:
        return False


def get_syntactic_validity(
    testcases: list[str],
    description: str,
    grammar: dict[str, list[str]],
) -> float:
    validness = [
        check_syntactic_validness(e, description, grammar)
        for e in testcases
    ]
    count = len(validness)
    return sum(validness) / count


def main(testcase_path: str, ground_truth_path: str) -> None:
    testcases = [
        e['testcase'] for e in jsonlines.open(testcase_path)]
    ground_truth_list = [
        e['grammar'] for e in jsonlines.open(ground_truth_path)]
    descriptions = [
        e['name'] for e in jsonlines.open(testcase_path)]

    validity_list = [
        get_syntactic_validity(testcase, description, ground_truth)
        for testcase, description, ground_truth
        in tqdm(zip(testcases, descriptions, ground_truth_list))
        if len(testcase) > 0
    ]
    grammar_based_validity_list = [
        1 if validity == 1 else 0
        for validity in validity_list
    ]
    count = len(validity_list)
    print(f'Ground Truth: {len(ground_truth_list)}')
    print(f'Problem with test case: {count}')
    print(f'Testcase-Based Syntactic_validity: {sum(validity_list)/count}')
    print(
        'Grammar-based Syntactic_validity:',
        sum(grammar_based_validity_list)/count
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth')
    args = parser.parse_args()

    main(args.testcase, args.ground_truth)
