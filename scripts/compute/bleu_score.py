import argparse
import os
from pathlib import Path
from statistics import mean
from typing import Optional, Iterable, Any

import jsonlines  # type: ignore
from sacrebleu import corpus_bleu as bleu_score  # type: ignore
from utils import sanitize  # type: ignore
from utils import normalize_grammar


def string_list_to_corpus(list_of_strings: list[str]) -> list[list[str]]:
    return [sum([e.split(" ") for e in list_of_strings], [])]


def compute_bleu(
    grammar: dict[str, list[str]], ground_truth: dict[str, list[str]]
) -> dict[str, float]:
    grammar_production = " ; ".join(grammar["productions"])
    ground_truth_production = " ; ".join(ground_truth["productions"])

    grammar_constraints = " ; ".join(grammar["constraints"])
    ground_truth_constraints = " ; ".join(ground_truth["constraints"])

    grammar_total = grammar_production + " ;; " + grammar_constraints
    ground_truth_total = (
        ground_truth_production + " ;; " + ground_truth_constraints
    )

    production_bleu = bleu_score(
        [ground_truth_production], [[grammar_production]]
    )
    constraints_bleu = bleu_score(
        [ground_truth_constraints], [[grammar_constraints]]
    )
    total_bleu = bleu_score([ground_truth_total], [[grammar_total]])
    return {
        "production": production_bleu.score,
        "constraints": constraints_bleu.score,
        "total": total_bleu.score,
    }


def main(grammar_path: Path) -> None:
    grammar_dataset = jsonlines.open(grammar_path)
    ref_dataset = jsonlines.open(
        Path(os.environ["GROUND_TRUTH_GRAMMAR_DIR"]) / grammar_path.name
    )

    prod_scores = []
    constr_scores = []
    total_scores = []

    for pred_data, ref_data in sanitize(zip(grammar_dataset, ref_dataset)):
        name = pred_data["name"]
        assert name == ref_data["name"]

        pred_grammar = pred_data["grammar"]
        ref_grammar = ref_data["grammar"]
        assert ref_grammar

        if pred_grammar is None:
            prod_scores.append(0.0)
            constr_scores.append(0.0)
            total_scores.append(0.0)
            continue

        pred_grammar = normalize_grammar(pred_grammar)
        ref_grammar = normalize_grammar(ref_grammar)
        bleu_results = compute_bleu(pred_grammar, ref_grammar)
        prod_bleu = bleu_results["production"]
        constr_bleu = bleu_results["constraints"]
        total_bleu = bleu_results["total"]

        prod_scores.append(prod_bleu)
        constr_scores.append(constr_bleu)
        total_scores.append(total_bleu)

    count = len(prod_scores)
    assert count == len(constr_scores)
    assert count == len(total_scores)

    print(f"total production-blue constraints-bleu total-bleu")
    print(
        f"{count} {mean(prod_scores)} {mean(constr_scores)} {mean(total_scores)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar", type=Path)
    args = parser.parse_args()

    main(args.grammar)
