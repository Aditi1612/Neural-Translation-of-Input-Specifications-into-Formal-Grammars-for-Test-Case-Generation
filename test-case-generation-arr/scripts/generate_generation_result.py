import argparse

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
    grammar: dict[str, list[str]],
) -> bool:
    try:
        validness = _check_syntactic_validness(testcase, grammar)
        return validness
    except timeout_decorator.TimeoutError:
        return False


def get_syntactic_validiness_list(
    testcases: list[str],
    grammar: dict[str, list[str]],
) -> list[bool]:
    validness = [check_syntactic_validness(e, grammar) for e in testcases]
    return validness


def main(testcase_path: str, ground_truth_path: str, output_path: str) -> None:
    testcases = [
        e['testcase'] for e in jsonlines.open(testcase_path)]
    ground_truth_list = [
        e['grammar'] for e in jsonlines.open(ground_truth_path)]

    validity_list = [
        get_syntactic_validiness_list(testcase, grammar)
        for testcase, grammar in tqdm(zip(testcases, ground_truth_list))
    ]
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(validity_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('testcase')
    parser.add_argument('ground_truth_grammar')
    parser.add_argument('output')
    args = parser.parse_args()

    main(args.testcase, args.ground_truth_grammar, args.output)
