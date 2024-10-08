import argparse
from typing import Optional

import jsonlines  # type: ignore
from sacrebleu import corpus_bleu as bleu_score  # type: ignore
from utils import get_filter_list
from utils import normalize_grammar


def string_list_to_corpus(list_of_strings: list[str]) -> list[list[str]]:
    return [sum([e.split(" ") for e in list_of_strings], [])]


def compute_bleu(
    grammar: dict[str, list[str]], ground_truth: dict[str, list[str]]
) -> tuple[float, float]:
    bleu_1 = bleu_score(
        [" ".join(ground_truth["productions"])],
        [[" ".join(grammar["productions"])]],
    )
    bleu_2 = bleu_score(
        [" ".join(ground_truth["constraints"])],
        [[" ".join(grammar["constraints"])]],
    )
    return bleu_1.score, bleu_2.score


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

    production_bleu = []
    constraint_bleu = []
    total_bleu = []

    num = len(grammar_dataset)
    filter_list = get_filter_list(filter_path_1, filter_path_2, num)
    grammar_dataset = [e for e, f in zip(grammar_dataset, filter_list) if f]
    ground_truth_dataset = [
        e for e, f in zip(ground_truth_dataset, filter_list) if f
    ]

    for ground_truth, grammar in zip(ground_truth_dataset, grammar_dataset):
        if grammar is None:
            production_bleu.append(0.0)
            constraint_bleu.append(0.0)
            total_bleu.append(0.0)
            continue

        ground_truth = normalize_grammar(ground_truth)
        grammar = normalize_grammar(grammar)

        bleu_1, bleu_2 = compute_bleu(grammar, ground_truth)
        production_bleu.append(bleu_1)
        constraint_bleu.append(bleu_2)
        total_bleu.append((bleu_1 + bleu_2) / 2)

    count = len(production_bleu)
    print(f"Total: {count}")
    production = sum(production_bleu) / count
    constraint = sum(constraint_bleu) / count
    both = sum(total_bleu) / count
    print(f"{production}, {constraint}, {both}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("grammar")
    parser.add_argument("ground_truth")
    parser.add_argument("--filter1")
    parser.add_argument("--filter2")
    args = parser.parse_args()

    main(args.grammar, args.ground_truth, args.filter1, args.filter2)
