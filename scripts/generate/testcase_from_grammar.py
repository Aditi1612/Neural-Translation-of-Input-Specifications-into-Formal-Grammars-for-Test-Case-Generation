import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import jsonlines  # type: ignore
import timeout_decorator  # type: ignore
from timeout_decorator.timeout_decorator import TimeoutError  # type: ignore
from tqdm import tqdm  # type: ignore

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg
# from utils import get_testcase_num_dict


SEED = 42
random.seed(SEED)


def get_testcase(
    ccfg: Ccfg,
    timeout: int,
    mode: str = "extreme",
) -> tuple[str, str]:
    """
    throw: Error
    """
    @timeout_decorator.timeout(timeout)
    def _generate_extreme() -> str:
        ccfg.extrememode = True
        ccfg.testmode = False
        return ccfg.generate()

    @timeout_decorator.timeout(timeout)
    def _generate_normal() -> str:
        ccfg.extrememode = True
        ccfg.testmode = False
        return ccfg.generate()

    @timeout_decorator.timeout(timeout)
    def _generate_test() -> str:
        ccfg.extrememode = False
        ccfg.testmode = True
        return ccfg.generate()

    if mode == "extreme":
        try:
            return _generate_extreme(), "extreme"
        except TimeoutError:
            pass

    if mode == "normal" or mode == "extreme":
        try:
            return _generate_normal(), "normal"
        except TimeoutError:
            pass

    return _generate_test(), "test"


def get_testcases(
    data: dict,
    k: int,
    timeout: int,
) -> Optional[tuple[list[str], list[str]]]:
    grammar = data["grammar"]

    if grammar is None:
        return None

    productions = grammar["productions"]
    constraints = grammar["constraints"]

    ccfg = Ccfg(productions, constraints, extrememode=True)

    num_extreme = k // 3
    num_normal = k // 3
    num_test = k - num_extreme - num_normal

    tuples = (
        [get_testcase(ccfg, timeout) for _ in range(num_extreme)]
        + [get_testcase(ccfg, timeout, mode="normal")
           for _ in range(num_normal)]
        + [get_testcase(ccfg, timeout, mode="test") for _ in range(num_test)]
    )
    return [t[0] for t in tuples], [t[1] for t in tuples]


def main(args: argparse.Namespace):
    grammar_path = Path(args.grammar_data)
    output_path = Path(args.output)
    # name = grammar_path.stem
    # num_dict = get_testcase_num_dict(name)

    with jsonlines.open(grammar_path, "r") as grammar_dataset:
        with jsonlines.open(output_path, "w") as writer:
            for data in tqdm(grammar_dataset):
                try:
                    # k = num_dict[data["name"]]
                    k = 30
                    pair = get_testcases(data, k, args.timeout)
                    if pair is None:
                        raise Exception("Grammar is None")

                    testcases, methods = pair
                    data.update({
                        "testcase": testcases,
                        "methods": methods,
                    })
                    writer.write(data)
                except Exception as e:
                    logging.warn(data["name"])
                    logging.warn(f"Error: {e}")
                    data.update({"error": str(e)})
                    writer.write(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar-data")
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()

    main(args)
