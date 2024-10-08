from pathlib import Path
import sys
import os

import jsonlines
from tqdm import tqdm


def main() -> None:
    raw_data_path = Path("raw-data/code_contests_train.jsonl")
    raw_data = jsonlines.open(raw_data_path, "r")
    data_dict = {}
    for data in tqdm(raw_data):
        data_dict[data["name"]] = data

    # list of problem names that have ground-truth grammar
    ground_truth_grammar_path = Path(
        f"data/grammar/ground-truth/{sys.argv[1]}.jsonl"
    )
    names = [e["name"] for e in jsonlines.open(ground_truth_grammar_path, "r")]

    code_contest = {
        "public": jsonlines.open(
            f"data/testcase/code-contest/public/{sys.argv[1]}.jsonl", "w"
        ),
        "private": jsonlines.open(
            f"data/testcase/code-contest/private/{sys.argv[1]}.jsonl", "w"
        ),
        "generated": jsonlines.open(
            f"data/testcase/code-contest/generated/{sys.argv[1]}.jsonl", "w"
        ),
    }

    for name in names:
        data = data_dict[name]
        for k, v in code_contest.items():
            testcases = data[f"{k}_tests"]["input"]
            output = data[f"{k}_tests"]["output"]
            new_data = {
                "name": name,
                "description": data["description"],
                "testcase": testcases,
                "output": output,
                "difficulty": data.get("difficulty", None),
                "time_limit": data.get("time_limit", None),
            }
            v.write(new_data)


if __name__ == "__main__":
    main()
