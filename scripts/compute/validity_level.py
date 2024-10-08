import argparse
import os
from pathlib import Path

import jsonlines  # type: ignore
from utils import sanitize


def main(generation_path: Path) -> None:
    generation_results = jsonlines.open(generation_path)
    levels = jsonlines.open(
        Path(os.environ["LEVEL_DIR"]) / generation_path.name
    )

    # Number of grammars that succeed to generate test cases
    count = 0
    level_to_validities: dict[int, list[float]] = {
        level: [] for level in range(1, 6)
    }

    for generation_result, level_data in sanitize(
        zip(generation_results, levels)
    ):
        count += 1

        assert generation_result["name"] == level_data["name"]
        level = level_data["max_level"]

        validities = level_to_validities[level]

        testcase_results = generation_result["results"]
        if len(testcase_results) == 0:
            continue
        num_parsable = sum(e["parsable"] for e in testcase_results)

        validity = num_parsable / len(testcase_results)
        validities.append(validity)

    level_to_elementwise_validity = {
        level: sum(validities) / len(validities) * 100 if validities else 0
        for level, validities in level_to_validities.items()
    }
    level_to_set_validity = {
        level: (
            sum(e == 1 for e in validities) / len(validities) * 100
            if validities
            else 0
        )
        for level, validities in level_to_validities.items()
    }

    print("total")
    print(" ".join([f"validity-{level}" for level in range(1, 6)]))
    print(count)
    print("used")
    print(
        " ".join(
            [str(len(valities)) for valities in level_to_validities.values()]
        )
    )
    print("element-wise")
    print(
        " ".join(
            [
                str(validity)
                for validity in level_to_elementwise_validity.values()
            ]
        )
    )
    print("set")
    print(
        " ".join([str(validity) for validity in level_to_set_validity.values()])
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-result", type=Path)
    args = parser.parse_args()
    main(args.generation_result)
