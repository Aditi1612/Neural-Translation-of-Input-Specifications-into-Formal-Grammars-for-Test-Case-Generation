import argparse
import os
from pathlib import Path
from statistics import mean
from typing import Optional, Iterable

import jsonlines  # type: ignore
from utils import sanitize  # type: ignore
from utils import normalize_grammar
from utils import GrammarData
from utils import Grammar


def compare_grammar(g1: Grammar, g2: Grammar) -> tuple[bool, bool]:
    production_matched = set(g1["productions"]) == set(g2["productions"])
    constraint_matched = set(g1["constraints"]) == set(g2["constraints"])
    return production_matched, constraint_matched


def main(grammar_path: Path) -> None:

    grammar_dataset = jsonlines.open(grammar_path)
    reference_dataset = jsonlines.open(
        Path(os.environ["GROUND_TRUTH_GRAMMAR_DIR"]) / grammar_path.name
    )

    production_matched_list: list[bool] = []
    constraint_matched_list: list[bool] = []

    for grammar_data, reference_data in sanitize(
        zip(grammar_dataset, reference_dataset)
    ):
        name = grammar_data["name"]
        assert name == reference_data["name"]

        grammar = grammar_data["grammar"]
        reference = reference_data["grammar"]

        assert reference
        if grammar is None:
            production_matched_list.append(False)
            constraint_matched_list.append(False)
            continue

        reference = normalize_grammar(reference)
        grammar = normalize_grammar(grammar)

        production_matched, constraint_matched = compare_grammar(
            grammar, reference
        )
        production_matched_list.append(production_matched)
        constraint_matched_list.append(constraint_matched)
    both_matched_list = [
        p and c
        for p, c in zip(production_matched_list, constraint_matched_list)
    ]

    total = len(production_matched_list)
    assert total == len(constraint_matched_list)
    assert total == len(both_matched_list)

    production = mean(production_matched_list)
    constraint = mean(constraint_matched_list)
    both = mean(both_matched_list)
    print("total production constraint both")
    print(f"{total} {production * 100} {constraint * 100} {both * 100}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar", type=Path)
    args = parser.parse_args()

    main(args.grammar)
