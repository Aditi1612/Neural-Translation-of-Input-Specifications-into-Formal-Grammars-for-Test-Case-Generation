import argparse
from pathlib import Path

import jsonlines  # type: ignore
import numpy as np
from utils import sanitize  # type: ignore[import-untyped]


def main(execution_summary_path: Path, generation_path: Path) -> None:

    summaries = jsonlines.open(execution_summary_path)
    generation_results = jsonlines.open(generation_path)

    count = 0
    effectiveness_list = []
    it = sanitize(zip(summaries, generation_results))
    for summary, generation_result in it:
        count += 1

        assert summary["name"] == generation_result["name"], summary["name"]

        testcase_summaries = summary["results"]
        testcase_generations = generation_result["results"]
        validities = [e["parsable"] for e in testcase_generations]
        assert len(testcase_summaries) == len(validities), summary["name"]

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

        # Sample at most 10 testcases
        if len(testcase_summaries) == 31:
            testcase_summaries = (
                testcase_summaries[0:1]
                + testcase_summaries[1:4]
                + testcase_summaries[11:14]
                + testcase_summaries[21:24]
            )
        elif len(testcase_summaries) == 30:
            testcase_summaries = (
                testcase_summaries[0:4]
                + testcase_summaries[10:13]
                + testcase_summaries[20:23]
            )
        else:
            m = min(10, len(testcase_summaries))
            testcase_summaries = testcase_summaries[:m]

        assert len(testcase_summaries) <= 10, len(testcase_summaries)

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

    args = parser.parse_args()
    main(args.execution_summary, args.generation_result)
