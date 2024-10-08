import argparse
from pathlib import Path

import jsonlines  # type: ignore
from utils import sanitize  # type: ignore[import-untyped]


def main(generation_path: Path) -> None:
    generation_results = jsonlines.open(generation_path)

    # Number of grammars that succeed to generate test cases
    count = 0
    validities = []
    for generation_result in sanitize(generation_results):
        count += 1
        testcase_results = generation_result["results"]
        if len(testcase_results) == 0:
            continue
        num_parsable = sum(e["parsable"] for e in testcase_results)

        validity = num_parsable / len(testcase_results)
        validities.append(validity)

    if len(validities) == 0:
        print("Error")
        return

    elementwise_validity = sum(validities) / len(validities)
    set_validity = sum(e == 1 for e in validities) / len(validities)
    print("total used element-wise set")
    print(
        count, len(validities), elementwise_validity * 100, set_validity * 100
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-result", type=Path)
    args = parser.parse_args()
    main(args.generation_result)
