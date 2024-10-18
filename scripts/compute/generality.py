import argparse
from pathlib import Path
from typing import Iterable
from statistics import mean

import jsonlines  # type: ignore
from utils import sanitize  # type: ignore
from utils import GenerationResult, ParsingResult


def main(
    generation_dataset: Iterable[GenerationResult],
    parsing_dataset: Iterable[ParsingResult],
) -> None:

    # Number of grammars that succeed to generate test cases
    generalities = []
    validities = []

    for generation_data, parsing_data in sanitize(
        zip(generation_dataset, parsing_dataset)
    ):
        parsing_results = parsing_data["results"]
        generation_results = generation_data["results"]

        assert parsing_results
        if not generation_results:
            generalities.append(0)
            validities.append(0)
            continue

        # Generality is the ratio of candidate-parsable ground-truth test cases
        validity = mean([e["parsable"] for e in generation_results])
        generality = mean([e["parsable"] for e in parsing_results])

        validities.append(validity)
        generalities.append(generality)

    assert generalities
    assert len(generalities) == len(validities)

    total = len(generalities)
    element_generality = mean(generalities)
    set_generality = mean([e == 1 for e in generalities])
    correctness = mean(
        [e1 == 1 and e2 == 1 for e1, e2 in zip(generalities, validities)]
    )

    print("total element-generality set-generality correctness")
    print(
        total, element_generality * 100, set_generality * 100, correctness * 100
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-result", type=jsonlines.open)
    parser.add_argument("--parsing-result", type=jsonlines.open)
    args = parser.parse_args()
    main(args.generation_result, args.parsing_result)
