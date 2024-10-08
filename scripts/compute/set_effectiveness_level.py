import argparse
import os
from pathlib import Path

import jsonlines  # type: ignore
import numpy as np
from utils import sanitize


def main(execution_summary_path: Path, generation_path: Path) -> None:

    summaries = jsonlines.open(execution_summary_path)
    generation_results = jsonlines.open(generation_path)
    levels = jsonlines.open(
        Path(os.environ["LEVEL_DIR"]) / execution_summary_path.name
    )

    count = 0
    level_to_effectiveness_list: dict[int, list[float]] = {
        level: [] for level in range(1, 6)
    }

    it = sanitize(zip(summaries, generation_results, levels))
    for summary, generation_result, level_data in it:
        count += 1

        assert summary["name"] == generation_result["name"]
        assert summary["name"] == level_data["name"]

        level = level_data["max_level"]

        effectiveness_list = level_to_effectiveness_list[level]

        testcase_summaries = summary["results"]
        testcase_generations = generation_result["results"]
        validities = [e["parsable"] for e in testcase_generations]
        assert len(testcase_summaries) == len(validities)

        # We consider the effectiveness of failed grammar as 0
        if not testcase_summaries:
            effectiveness_list.append(0)
            continue

        # We consider the effectiveness of an invalid set of testcase as 0
        if not all(validities):
            effectiveness_list.append(0)
            continue

        # Initialize expectation of incorrect results
        total_incorrect_results = [True] * len(
            testcase_summaries[0]["incorrect_results"]
        )

        for testcase_summary in testcase_summaries:
            incorrect_results = testcase_summary["incorrect_results"]
            assert incorrect_results

            for i, incorrect_result in enumerate(incorrect_results):
                total_incorrect_results[i] &= incorrect_result

        effectiveness = 1 - sum(total_incorrect_results) / len(
            total_incorrect_results
        )
        effectiveness_list.append(effectiveness)

    level_to_effectiveness = {
        level: np.average(effectiveness_list) * 100 if effectiveness_list else 0
        for level, effectiveness_list in level_to_effectiveness_list.items()
    }

    level_to_number = {
        level: len(effectiveness_list)
        for level, effectiveness_list in level_to_effectiveness_list.items()
    }

    print("total")
    print(count)
    print(" ".join([f"effectiveness-{level}" for level in range(1, 6)]))
    print(" ".join([str(number) for number in level_to_number.values()]))
    print(
        " ".join(
            [
                str(effectiveness)
                for effectiveness in level_to_effectiveness.values()
            ]
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-summary", type=Path)
    parser.add_argument("--generation-result", type=Path)

    args = parser.parse_args()
    main(args.execution_summary, args.generation_result)
