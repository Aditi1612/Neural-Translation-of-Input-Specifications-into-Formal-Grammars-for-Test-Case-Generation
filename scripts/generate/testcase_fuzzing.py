import argparse
import math
import random
import re
import string
from typing import Union

import jsonlines
from tqdm import tqdm


class Fuzzer:
    def __init__(
        self, fuzzing_rate=0.3, numeric_addition_bound=1, num_of_result=10
    ):
        self.fuzzing_rate = fuzzing_rate
        self.numeric_addition_bound = numeric_addition_bound
        self.num_of_result = num_of_result

    def __call__(self, test_cases: list[str]) -> list[list[str]]:
        res = []
        for idx, test_case in enumerate(test_cases):
            test_case = test_case.rstrip()
            tokens_origin = re.split(r"[ \n\{\},]", test_case)
            spliters = re.findall(r"[ \n\{\},]", test_case)
            num_of_tokens = len(tokens_origin)
            num_of_sample = math.ceil(num_of_tokens * self.fuzzing_rate)
            weight = [1] * len(tokens_origin)

            weight[0] = (
                0
                if (
                    re.fullmatch(r"[0-9]*", tokens_origin[0])
                    and num_of_tokens > 1
                )
                else 1
            )

            mutateds = []
            for _ in range(self.num_of_result):
                tokens = tokens_origin[:]
                targets = random.choices(
                    range(num_of_tokens), weights=weight, k=num_of_sample
                )
                mutated = ""
                for target_idx in targets:
                    target = tokens[target_idx]
                    if not target:
                        continue
                    if re.fullmatch(r"[-+]?[0-9]+", target):
                        addition_value: Union[int, float] = 0
                        while not addition_value:
                            addition_value = random.randint(
                                self.numeric_addition_bound * -1,
                                self.numeric_addition_bound,
                            )
                        tokens[target_idx] = str(int(target) + addition_value)
                    elif re.fullmatch(r"[-+]?([0-9]+\.[0-9]+|[0-9]+)", target):
                        addition_value = 0.0
                        while not addition_value:
                            addition_value = random.uniform(
                                self.numeric_addition_bound * -1,
                                self.numeric_addition_bound,
                            )
                        tokens[target_idx] = str(float(target) + addition_value)
                    elif re.fullmatch(r"[a-zA-Z]*", target):
                        str_list = list(self.get_str_list(target))

                        remove_index = random.choice(range(len(target)))
                        str_list.remove(target[remove_index])
                        target = list(target)
                        target[remove_index] = random.choice(str_list)
                        tokens[target_idx] = "".join(target)
                    else:
                        change = list(target)
                        while (
                            "".join(change) == target
                            and len(target) > 1
                            and len(set(change)) > 1
                        ):
                            random.shuffle(change)
                        tokens[target_idx] = "".join(change)

                mutated += tokens[0]
                for token, spliter in zip(tokens[1:], spliters):
                    mutated += spliter
                    mutated += token
                mutateds.append(mutated)
            res.append(mutateds)

        return res

    def get_str_list(self, token: str):
        if token == token.lower():
            return string.ascii_lowercase
        elif token == token.upper():
            return string.ascii_uppercase
        else:
            return string.ascii_letters


def update(fuzzer: Fuzzer, data: dict) -> dict:
    testcases = data["testcase"]
    fuzzed: list[str] = sum(fuzzer(testcases), [])
    num = min(10, len(fuzzed))
    data["testcase"] = random.sample(fuzzed, num)
    return data


def main(args: argparse.Namespace):
    input_path = args.input
    output_path = args.output
    fuzzer = Fuzzer()

    with jsonlines.open(input_path, "r") as input_dataset:
        with jsonlines.open(output_path, "w") as writer:
            writer.write_all(
                map(lambda e: update(fuzzer, e), tqdm(input_dataset))
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()

    main(args)
