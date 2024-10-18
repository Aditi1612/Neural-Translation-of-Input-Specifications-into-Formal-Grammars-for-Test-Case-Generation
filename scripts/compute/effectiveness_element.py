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
        assert summary_data["name"] == generation_data["name"]

        summary_results = summary_data["results"]
        generation_results = generation_data["results"]
        validities = [e["parsable"] for e in generation_results]
        assert len(summary_results) == len(validities)

        # We consider the effectiveness of failed grammar as 0
        if not summary_results:
            effectiveness_list.append(0.0)
            continue

        # Calculate the effectiveness of each test case
        testcase_effectivenesses = []
        for testcase_summary, valid in zip(summary_results, validities):
            incorrect_results = testcase_summary["incorrect_results"]
            assert len(incorrect_results) > 0

            # We consider the effectiveness of an invalid test case as 0
            if not valid:
                testcase_effectivenesses.append(0)
                continue
            testcase_effectiveness = 1 - sum(incorrect_results) / len(
                incorrect_results
            )
            testcase_effectivenesses.append(testcase_effectiveness)

        # Average effectiveness of a grammar
        effectiveness = sum(testcase_effectivenesses) / len(
            testcase_effectivenesses
        )
        effectiveness_list.append(effectiveness)

    if not use_level:
        total = len(effectiveness_list)
        average_effectiveness = np.average(effectiveness_list)
        std = np.std(effectiveness_list)

        print("total element-effectiveness std")
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

            print("level total element-effectiveness std")
            print(level, total, average_effectiveness * 100, std * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-summary", type=Path)
    parser.add_argument("--generation-result", type=Path)
    parser.add_argument("--level", action="store_true")

    args = parser.parse_args()
    main(args.execution_summary, args.generation_result, args.level)
