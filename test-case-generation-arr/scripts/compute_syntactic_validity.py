import argparse
from typing import Optional

import jsonlines  # type: ignore
import numpy as np

from utils import get_filter_list


def main(
    generation_path: str,
    filter_path_1: Optional[str],
    filter_path_2: Optional[str]
) -> None:
    generation_results = list(jsonlines.open(generation_path))
    num = len(generation_results)
    filter_list = get_filter_list(filter_path_1, filter_path_2, num)
    generation_results = [
        e for e, f in zip(generation_results, filter_list) if f]

    validity_list = [sum(e) / len(e) for e in generation_results if len(e) > 0]
    grammar_based_validity_list = [
        all(e) for e in generation_results if len(e) > 0]

    count = len(validity_list)
    grammar_based = np.average(grammar_based_validity_list) * 100
    testcases_based = np.average(validity_list) * 100

    std_1 = np.std(grammar_based_validity_list) * 100
    std_2 = np.std(validity_list) * 100

    print("total, grammar-based, testcases-based")
    print(f"{count} & {grammar_based:.2f} & {testcases_based:.2f} {{\small($\\pm {std_2:.2f}$)}} & \\\\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('generation_result')
    parser.add_argument('--filter1')
    parser.add_argument('--filter2')
    args = parser.parse_args()

    main(args.generation_result, args.filter1, args.filter2)
