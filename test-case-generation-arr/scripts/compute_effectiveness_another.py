import argparse
import sys
from pathlib import Path
from typing import Optional, cast

import jsonlines  # type: ignore

from utils import get_mode
from utils import get_filter_list


def get_num_of_incorrect_solution(name: str) -> int:
    incorrect_solution_prefix = Path('./data/solutions/incorrect_solutions')
    incorrect_solution_dir = incorrect_solution_prefix / name
    return len(list(incorrect_solution_dir.glob('*.py')))


def get_distinguished_list(
    execution: dict[str, list[Optional[str]]],
    num: int
) -> Optional[list[bool]]:
    _correct_results = execution['correct']
    correct_results = cast(
        list[str], [e for e in _correct_results if e is not None])
    if len(correct_results) == 0:
        return None
    answer = get_mode(correct_results)[0]
    incorrect_results = execution['incorrect']
    distinguished = [e != answer for e in incorrect_results]
    return distinguished


def get_ratio_of_distinguished(
    distinguished_lists: list[list[bool]]
) -> float:
    distinguished = []
    for i in range(len(distinguished_lists[0])):
        distinguished.append(any(e[i] for e in distinguished_lists))
    return sum(distinguished) / len(distinguished)


def get_average_effectiveness(
    distinguished_lists: list[list[bool]]
) -> float:
    effectiveness_list = [
        sum(distinguished) / len(distinguished)
        for distinguished in distinguished_lists
    ]
    return sum(effectiveness_list) / len(effectiveness_list)


def main(
    executions_path: str,
    generation_path: str,
    testcase_path: str,
    filter_path_1: Optional[str],
    filter_path_2: Optional[str]
):
    executions_list = list(jsonlines.open(executions_path, 'r'))
    generation_list = list(jsonlines.open(generation_path, 'r'))
    names = [e['name'] for e in jsonlines.open(testcase_path, 'r')]

    num = len(executions_list)
    filter_list = get_filter_list(filter_path_1, filter_path_2, num)
    executions_list = [e for e, f in zip(executions_list, filter_list) if f]
    generation_list = [e for e, f in zip(generation_list, filter_list) if f]
    names = [e for e, f in zip(names, filter_list) if f]

    effectiveness_list = []
    ratio_of_distinguished_list = []
    for executions, generations, name in zip(
        executions_list,
        generation_list,
        names
    ):
        num = min(get_num_of_incorrect_solution(name), 10)

        # 1. Remove problem with no incorrect solution from sample
        if num == 0:
            continue

        # 3. if a testcase is syntactically invalid, ignore it.
        # 4. if every execution on correct answer is None, ignore it.
        _distinguished_lists = [
            get_distinguished_list(execution, num)
            for execution, generation in zip(executions, generations)
            if generation
        ]
        distinguished_lists = cast(
            list[list[bool]],
            [e for e in _distinguished_lists if e is not None]
        )

        if len(distinguished_lists) == 0:
            print(f'Name: {name}', file=sys.stderr)
            continue

        effectiveness = get_average_effectiveness(distinguished_lists)
        distinguished = get_ratio_of_distinguished(distinguished_lists)

        assert distinguished is not None
        effectiveness_list.append(effectiveness)
        ratio_of_distinguished_list.append(distinguished)

    count = len(effectiveness_list)
    assert count == len(ratio_of_distinguished_list)
    # print(f'Total: {count}')
    # print(f'Effectiveness: {sum(effectiveness_list)/count}')
    # print(f'Average ratio: {sum(ratio_of_distinguished_list)/count}')
    effectiveness = sum(effectiveness_list) / count
    ratio_of_distinguished = sum(ratio_of_distinguished_list) / count
    print(f"{count}, {effectiveness * 100}, {ratio_of_distinguished * 100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('executions')
    parser.add_argument('generation')
    parser.add_argument('testcase')
    parser.add_argument('--filter1')
    parser.add_argument('--filter2')
    args = parser.parse_args()
    main(
        args.executions,
        args.generation,
        args.testcase,
        args.filter1,
        args.filter2
    )
