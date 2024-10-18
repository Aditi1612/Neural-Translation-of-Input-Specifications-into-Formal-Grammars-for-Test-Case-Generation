import argparse
from pathlib import Path
from typing import Iterable, Any, Optional

import jsonlines  # type: ignore
import numpy as np
from utils import sanitize  # type: ignore
from utils import ExecutionSummary
from utils import GenerationResult
from utils import split_with_level


def main(
    summary_path: Path,
    generation_path: Path,
    use_level: bool,
) -> None:
    assert summary_path.stem == generation_path.stem

    summary_dataset = jsonlines.open(summary_path)
    generation_dataset = jsonlines.open(generation_path)

    effectiveness_list = []
    for summary_data, generation_data in sanitize(
        zip(summary_dataset, generation_dataset)
    ):
        name = summary_data["name"]
        assert name == generation_data["name"], name

        summary_results = summary_data["results"]
        generation_results = generation_data["results"]
        validities = [e["parsable"] for e in generation_results]
        assert len(summary_results) == len(validities), name

        # We consider the effectiveness of failed grammar as 0
        if len(summary_results) == 0:
            effectiveness_list.append(0.0)
            continue

        # We consider the effectiveness of an invalid set of test cases as 0
        if not all(validities):
            effectiveness_list.append(0.0)
            continue

        # Initialize expectation of incorrect results
        total_incorrect_results = [True] * len(
            summary_results[0]["incorrect_results"]
        )

        # Select at most 10 test cases
        if len(summary_results) == 31:
            summary_results = (
                summary_results[0:1]
                + summary_results[1:4]
                + summary_results[11:14]
                + summary_results[21:24]
            )
        elif len(summary_results) == 30:
            summary_results = (
                summary_results[0:4]
                + summary_results[10:13]
                + summary_results[20:23]
            )
        else:
            m = min(10, len(summary_results))
            summary_results = summary_results[:m]

        assert len(summary_results) <= 10, len(summary_results)

        for testcase_summary in summary_results:
            incorrect_results = testcase_summary["incorrect_results"]
            assert len(incorrect_results) > 0

            for i, incorrect_result in enumerate(incorrect_results):
                total_incorrect_results[i] &= incorrect_result

        effectiveness = 1 - sum(total_incorrect_results) / len(
            total_incorrect_results
        )
        effectiveness_list.append(effectiveness)

    if not use_level:
        total = len(effectiveness_list)
        average_effectiveness = np.average(effectiveness_list)
        std = np.std(effectiveness_list)

        print("total set-effectiveness std")
        print(total, average_effectiveness * 100, std * 100)
    else:
        level_name = summary_path.stem
        level_to_effectiveness_list = split_with_level(
            effectiveness_list, level_name
        )
        for level, effectiveness_list in level_to_effectiveness_list.items():
            if not effectiveness_list:
                print(level, 0, 0, 0)
                continue

            total = len(effectiveness_list)
            average_effectiveness = np.average(effectiveness_list)
            std = np.std(effectiveness_list)

            print("level total set-effectiveness std")
            print(level, total, average_effectiveness * 100, std * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-summary", type=Path)
    parser.add_argument("--generation-result", type=Path)
    parser.add_argument("--level", action="store_true")

    args = parser.parse_args()
    main(args.execution_summary, args.generation_result, args.level)
