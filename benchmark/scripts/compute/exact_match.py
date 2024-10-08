import argparse
from typing import Optional

import jsonlines  # type: ignore
from utils import get_filter_list
from utils import normalize_grammar


def compare_grammar(
    grammar: dict[str, list[str]], ground_truth: dict[str, list[str]]
) -> tuple[bool, bool]:
    production_matched = False
    constraint_matched = False

    production_matched = set(ground_truth["productions"]) == set(
        grammar["productions"]
    )
    constraint_matched = set(ground_truth["constraints"]) == set(
        grammar["constraints"]
    )
    return production_matched, constraint_matched


def main(
    grammar_path: str,
    ground_truth_path: str,
    filter_path_1: Optional[str],
    filter_path_2: Optional[str],
) -> None:
    grammar_dataset = list(
        map(lambda e: e["grammar"], jsonlines.open(grammar_path))
    )
    ground_truth_dataset = list(
        map(lambda e: e["grammar"], jsonlines.open(ground_truth_path))
    )

    num = len(grammar_dataset)
    filter_list = get_filter_list(filter_path_1, filter_path_2, num)
    grammar_dataset = [e for e, f in zip(grammar_dataset, filter_list) if f]
    ground_truth_dataset = [
        e for e, f in zip(ground_truth_dataset, filter_list) if f
    ]

    production_matched_list: list[bool] = []
    constraint_matched_list: list[bool] = []
    matched_list: list[bool] = []

    for ground_truth, grammar in zip(ground_truth_dataset, grammar_dataset):
        if grammar is None:
            production_matched_list.append(False)
            constraint_matched_list.append(False)
            matched_list.append(False)
            continue

        ground_truth = normalize_grammar(ground_truth)
        grammar = normalize_grammar(grammar)

        production_matched, constraint_matched = compare_grammar(
            grammar, ground_truth
        )
        matched = production_matched and constraint_matched
        production_matched_list.append(production_matched)
        constraint_matched_list.append(constraint_matched)
        matched_list.append(matched)

    count = len(production_matched_list)
    print(f"Total: {count}")
    production = sum(production_matched_list) / count * 100
    constraint = sum(constraint_matched_list) / count * 100
    both = sum(matched_list) / count * 100
    print(f"{production}, {constraint}, {both}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grammar")
    parser.add_argument("ground_truth")
    parser.add_argument("--filter1")
    parser.add_argument("--filter2")
    args = parser.parse_args()

    main(args.grammar, args.ground_truth, args.filter1, args.filter2)
