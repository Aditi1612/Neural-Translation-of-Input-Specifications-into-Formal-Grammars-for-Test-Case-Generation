"""Generate testcases from grammar"""

import argparse
import logging
from pathlib import Path
import random
from typing import Any, Optional

import jsonlines  # type: ignore
import timeout_decorator  # type: ignore
from timeout_decorator.timeout_decorator import TimeoutError  # type: ignore
from tqdm import tqdm  # type: ignore

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg

SEED = 42
random.seed(SEED)


def get_testcase(
    ccfg: Ccfg,
    timeout: int,
    min_degree: int,
    k: int,
) -> list[tuple[str, int]]:
    """
    throw: Error
    """

    if min_degree == -1:
        assert k == 1
        return [(ccfg.generate(degree=-1), -1)]

    @timeout_decorator.timeout(timeout)
    def _generate(degree: int) -> str:
        return ccfg.generate(degree=degree)

    testcases: list[tuple[str, int]] = []
    degree = min_degree

    for _ in range(k):
        while True:
            try:
                testcases.append((_generate(degree), degree))
                break
            except TimeoutError as e:
                if degree >= 2:
                    raise e
                degree += 1

    return testcases


def get_testcases(
    data: dict[str, Any],
    k: int,
    timeout: int,
) -> Optional[tuple[list[str], list[int]]]:
    grammar = data["grammar"]

    if grammar is None:
        return None

    productions = grammar["productions"]
    constraints = grammar["constraints"]

    ccfg = Ccfg(productions, constraints)

    try:
        tuples = get_testcase(ccfg, timeout, -1, 1)
    except TimeoutError:
        tuples = []

    tuples += get_testcase(ccfg, timeout, 2, k)
    tuples += get_testcase(ccfg, timeout, 1, k)
    tuples += get_testcase(ccfg, timeout, 0, k)

    return [t[0] for t in tuples], [t[1] for t in tuples]


def main(grammar_path: Path, output_path: Path, k: int, timeout: int) -> None:

    with jsonlines.open(grammar_path, "r") as grammar_dataset:
        with jsonlines.open(output_path, "w") as writer:
            for data in tqdm(grammar_dataset):
                try:
                    pair = get_testcases(data, k, timeout)
                    if pair is None:
                        raise ValueError("Grammar is None")

                    testcases, methods = pair
                    data.update(
                        {
                            "testcase": testcases,
                            "methods": methods,
                        }
                    )
                    writer.write(data)
                except Exception as e:  # pylint: disable=broad-except
                    logging.warning(data["name"])
                    logging.warning("Error: %s", str(e))
                    data.update({"error": str(e)})
                    writer.write(data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar-data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()

    main(args.grammar_data, args.output, args.num, args.timeout)
