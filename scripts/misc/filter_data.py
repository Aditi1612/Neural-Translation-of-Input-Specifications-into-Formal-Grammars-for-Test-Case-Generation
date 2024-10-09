"""Filter data by name."""

import argparse
from pathlib import Path

import jsonlines


def main(input_path: Path, output_path: Path, filter_path: Path) -> None:
    """Main function."""

    input_file = jsonlines.open(input_path)
    filter_file = jsonlines.open(filter_path)
    output_file = jsonlines.open(output_path, "w")

    names = {filter_data["name"] for filter_data in filter_file}

    for input_data in input_file:
        if input_data["name"] in names:
            output_file.write(input_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filter", type=Path, required=True)
    args = parser.parse_args()
    main(args.input, args.output, args.filter)
