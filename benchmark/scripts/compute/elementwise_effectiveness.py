import argparse
from pathlib import Path

import jsonlines
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

        assert summary["name"] == generation_result["name"]

        testcase_summaries = summary["results"]
        testcase_generations = generation_result["results"]
        validities = [e["parsable"] for e in testcase_generations]
        assert len(testcase_summaries) == len(validities)

        # We consider the effectiveness of failed grammar as 0
        if len(testcase_summaries) == 0:
            effectiveness_list.append(0.0)
            continue

        testcase_effectivenesses = []
        it2 = zip(testcase_summaries, validities)

        for testcase_summary, valid in it2:
            incorrect_results = testcase_summary["incorrect_results"]
            assert len(incorrect_results) > 0

            # We consider the effectiveness of an invalid testcase as 0
            if not valid:
                testcase_effectivenesses.append(0)
                continue
            testcase_effectiveness = 1 - sum(incorrect_results) / len(
                incorrect_results
            )
            testcase_effectivenesses.append(testcase_effectiveness)

        # Average effectiveness of the grammar
        effectiveness = sum(testcase_effectivenesses) / len(
            testcase_effectivenesses
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
