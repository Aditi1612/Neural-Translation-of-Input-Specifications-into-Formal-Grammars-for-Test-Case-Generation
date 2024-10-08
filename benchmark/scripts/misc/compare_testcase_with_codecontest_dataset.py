import argparse
import os
from pathlib import Path

import jsonlines


def main(testcase_path: Path) -> None:
    target_file = jsonlines.open(testcase_path)
    public_file = jsonlines.open(
        Path(os.environ["PUBLIC_TESTCASE_DIR"]) / testcase_path.name
    )
    private_file = jsonlines.open(
        Path(os.environ["PRIVATE_TESTCASE_DIR"]) / testcase_path.name
    )

    overlapped: list[float] = []
    for target_object, public_object, private_object in zip(
        target_file, public_file, private_file
    ):
        name = target_object["name"]
        description = target_object["description"]
        assert name == public_object["name"], f"{name} != {public_object['name']}"
        assert name == private_object["name"], f"{name} != {private_object['name']}"

        source_testcases = (
            public_object["testcase"] + private_object["testcase"]
        )
        target_testcases = target_object.get("testcase", [])

        if len(target_testcases) == 0:
            continue

        overlapped_testcases = [
            testcase
            for testcase in target_testcases
            if testcase in source_testcases
        ]
        count = len(overlapped_testcases)
        if count > 0:
            print(f"Name: {name}")
            print(f"Description: {description}")
            print(f"Overlapped: {overlapped_testcases}")
        overlapped.append(count / len(target_testcases))
    print(f"Total: {len(overlapped)}")
    print(f"Overlapped: {sum(overlapped)/len(overlapped) * 100} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testcase", type=Path)
    args = parser.parse_args()
    main(args.testcase)
