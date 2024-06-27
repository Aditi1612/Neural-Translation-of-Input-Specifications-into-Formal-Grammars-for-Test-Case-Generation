import argparse
from typing import Optional

import jsonlines  # type: ignore

from utils import get_filter_list


def main(
    parsing_path: str,
    generation_path: str,
    filter_path_1: Optional[str],
    filter_path_2: Optional[str],
) -> None:

    parsing_results = list(jsonlines.open(parsing_path))
    generation_results = list(jsonlines.open(generation_path))

    num = len(parsing_results)
    filter_list = get_filter_list(filter_path_1, filter_path_2, num)
    parsing_results = [e for e, f in zip(parsing_results, filter_list) if f]
    generation_results = [
        e for e, f in zip(generation_results, filter_list) if f]

    generality_list = [
        sum(p) / len(p)
        for (g, p) in zip(generation_results, parsing_results)
        if len(g) > 0
    ]
    count = len(generality_list)
    print(f'Total: {count}')
    print(f'Generality: {sum(generality_list)/count * 100}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('parsing_result')
    parser.add_argument('generation_result')
    parser.add_argument('--filter1')
    parser.add_argument('--filter2')
    args = parser.parse_args()

    main(
        args.parsing_result,
        args.generation_result,
        args.filter1,
        args.filter2
    )
