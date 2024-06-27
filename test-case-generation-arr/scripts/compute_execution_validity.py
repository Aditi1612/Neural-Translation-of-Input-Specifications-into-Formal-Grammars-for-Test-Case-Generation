import argparse
import sys
from typing import Optional

import jsonlines  # type: ignore

from utils import get_filter_list


def get_execution_validness(
    execution: dict[str, list[Optional[str]]],
    name: str,
    testcase: str
) -> bool:
    correct_results = execution['correct']
    if len(correct_results) == 0:
        return False

    validness = None not in correct_results and len(set(correct_results)) == 1
    if (
        not validness and len(testcase) < 100
        and sum(len(e) for e in correct_results if e is not None) < 100
    ):
        print(f'Name: {name}', file=sys.stderr)
        print(f'Testcase: {testcase}', file=sys.stderr)
        print(f'Correct: {correct_results}', file=sys.stderr)
    return validness


def get_execution_validity(
    executions: list[dict[str, list[Optional[str]]]],
    name: str,
    testcases: list[str]
) -> float:
    validness_list = [
        get_execution_validness(execution, name, testcase)
        for execution, testcase in zip(executions, testcases)
    ]
    count = len(validness_list)
    return sum(validness_list) / count


def main(
    executions_path: str,
    testcases_path: str,
    filter_path_1: Optional[str],
    filter_path_2: Optional[str]
):
    executions_list = list(jsonlines.open(executions_path, 'r'))
    testcases_list = list(jsonlines.open(testcases_path, 'r'))

    num = len(executions_list)
    filter_list = get_filter_list(filter_path_1, filter_path_2, num)
    executions_list = [e for e, f in zip(executions_list, filter_list) if f]
    testcases_list = [e for e, f in zip(testcases_list, filter_list) if f]

    validity_list = []

    for executions, testcases in zip(executions_list, testcases_list):
        if len(executions) == 0:
            continue
        validity = get_execution_validity(
            executions, testcases["name"], testcases["testcase"])
        validity_list.append(validity)

    count = len(validity_list)
    validity_per_problem = [e == 1.0 for e in validity_list]
    print(f'Total: {count}')
    print(f'Execution_validity: {sum(validity_list)/count * 100}')
    print(f'Problem_validity: {sum(validity_per_problem)/count * 100}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('executions')
    parser.add_argument('testcases')
    parser.add_argument('--filter')
    args = parser.parse_args()
    main(args.executions, args.testcases, args.filter)
