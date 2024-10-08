import argparse
from typing import Optional

import jsonlines
from utils import get_filter_list  # type: ignore[import-untyped]
from utils import normalize_constraints
from utils import normalize_grammar
from utils import normalize_productions


def main(
    grammar_path: str,
    ground_truth_path: str,
    filter_path_1: Optional[str],
    filter_path_2: Optional[str],
) -> None:
    grammar_candidates_dataset = list(
        map(lambda e: e["grammar_candidates"], jsonlines.open(grammar_path))
    )
    ground_truth_dataset = list(
        map(lambda e: e["grammar"], jsonlines.open(ground_truth_path))
    )

    num = len(grammar_candidates_dataset)
    filter_list = get_filter_list(filter_path_1, filter_path_2, num)
    grammar_candidates_dataset = [
        e for e, f in zip(grammar_candidates_dataset, filter_list) if f
    ]
    ground_truth_dataset = [
        e for e, f in zip(ground_truth_dataset, filter_list) if f
    ]

    production_matched_list: list[bool] = []
    constraint_matched_list: list[bool] = []
    matched_list: list[bool] = []

    for ground_truth, grammar_candidates in zip(
        ground_truth_dataset, grammar_candidates_dataset
    ):
        if ground_truth is None:
            continue

        if grammar_candidates is None:
            production_matched_list.append(False)
            constraint_matched_list.append(False)
            matched_list.append(False)
            continue

        productions = [
            set(normalize_productions(e))
            for e in grammar_candidates["productions"]
        ]
        constraints = [
            set(normalize_constraints(e))
            for e in grammar_candidates["constraints"]
        ]
        ground_truth = normalize_grammar(ground_truth)

        production_matched = False
        constraint_matched = False
        if set(ground_truth["productions"]) in productions:
            production_matched = True

        if set(ground_truth["constraints"]) in constraints:
            constraint_matched = True

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
