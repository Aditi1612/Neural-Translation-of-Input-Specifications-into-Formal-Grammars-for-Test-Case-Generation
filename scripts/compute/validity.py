import argparse
from pathlib import Path
from typing import Iterable, Optional
from statistics import mean

import jsonlines  # type: ignore
from utils import sanitize  # type: ignore
from utils import GenerationResult
from utils import split_with_level


def main(generation_path: Path, use_level: bool) -> None:
    generation_dataset = jsonlines.open(generation_path)

    # Number of grammars that succeed to generate test cases
    well_definednesses = []
    validities = []

    for generation_data in sanitize(generation_dataset):
        generation_results = generation_data["results"]
        if not generation_results:
            well_definednesses.append(0)
            validities.append(0)
            continue

        well_definednesses.append(1)
        # Validity is the ratio of the number of parsable test cases to the number of test cases
        validity = mean([e["parsable"] for e in generation_results])
        validities.append(validity)

    assert validities

    if not use_level:
        element_validity = mean(validities) * 100
        set_validity = mean([e == 1 for e in validities]) * 100
        assert len(validities) == len(well_definednesses)

        print("total well-defined element-validity set-validity")
        print(
            len(validities),
            sum(well_definednesses),
            element_validity,
            set_validity,
        )
    else:
        level_name = generation_path.stem
        level_to_well_definednesses = split_with_level(
            well_definednesses, level_name
        )
        level_to_validities = split_with_level(validities, "test")
        for level in range(1, 6):
            well_definednesses = level_to_well_definednesses[level]
            validities = level_to_validities[level]
            assert len(validities) == len(well_definednesses)

            if not validities:
                continue

            element_validity = mean(validities) * 100
            set_validity = mean([e == 1 for e in validities]) * 100
            print("level total well-defined element-validity set-validity")
            print(
                level,
                len(validities),
                sum(well_definednesses),
                element_validity,
                set_validity,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-result", type=Path)
    parser.add_argument("--level", action="store_true")
    args = parser.parse_args()
    main(args.generation_result, args.level)
