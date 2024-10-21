import argparse
from pathlib import Path

import jsonlines  # type: ignore
import numpy as np
from utils import sanitize  # type: ignore


def main(
    execution_summary_path: Path, generation_path: Path, degree: int
) -> None:

    summaries = jsonlines.open(execution_summary_path)
    generation_results = jsonlines.open(generation_path)

    count = 0
    effectiveness_list = []
    it = sanitize(zip(summaries, generation_results))
    for summary, generation_result in it:
        count += 1

        assert summary["name"] == generation_result["name"]

        testcase_summaries = summary["results"]
        testcase_generations = generation_result["results"]
        validities = [e["parsable"] for e in testcase_generations]
        assert len(testcase_summaries) == len(validities)

        # We consider the effectiveness of failed grammar as 0
        if len(testcase_summaries) == 0:
            effectiveness_list.append(0.0)
            continue

        # We consider the effectiveness of an invalid set of testcase as 0
        if not all(validities):
            effectiveness_list.append(0.0)
            continue

        # Initialize expectation of incorrect results
        total_incorrect_results = [True] * len(
            testcase_summaries[0]["incorrect_results"]
        )
        if len(testcase_summaries) == 31:
            testcase_summaries = testcase_summaries[1 : (degree + 1) * 10 + 1]
        elif len(testcase_summaries) == 30:
            testcase_summaries = testcase_summaries[0 : (degree + 1) * 10]
        else:
            assert False

        for testcase_summary in testcase_summaries:
            incorrect_results = testcase_summary["incorrect_results"]
            assert len(incorrect_results) > 0

            for i, incorrect_result in enumerate(incorrect_results):
                total_incorrect_results[i] &= incorrect_result

        effectiveness = 1 - sum(total_incorrect_results) / len(
            total_incorrect_results
        )
        effectiveness_list.append(effectiveness)

    effectiveness = float(np.average(effectiveness_list))
    std = np.std(effectiveness_list)

    print("total used effectiveness std")
    print(count, len(effectiveness_list), effectiveness * 100, std * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-summary", type=Path)
    parser.add_argument("--generation-result", type=Path)
    parser.add_argument("--degree", type=int)

    args = parser.parse_args()
    main(args.execution_summary, args.generation_result, args.degree)
