import argparse
import logging
from pathlib import Path

import jsonlines  # type: ignore
from timeout_decorator import timeout  # type: ignore
from timeout_decorator import TimeoutError  # type: ignore
from tqdm import tqdm  # type: ignore

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg


@timeout(1)
def generate(ccfg: Ccfg) -> str:
    return ccfg.generate()


def update(data: dict, extreme: bool) -> dict:
    grammar = data["grammar"]

    if grammar is None:
        data["testcase"] = []
        return data

    productions = grammar["productions"]
    constraints = grammar["constraints"]

    try:
        ccfg = Ccfg(productions, constraints, extrememode=extreme)
    except Exception:
        data["testcase"] = []
        return data

    testcase: list[str] = []
    max_iter = 100
    for i in range(max_iter):
        if len(testcase) >= 10:
            break
        try:
            testcase.append(generate(ccfg))
        except TimeoutError:
            logging.warning("TimeoutError")
            if ccfg.extrememode:
                ccfg.extrememode = False
            elif not ccfg.testmode:
                ccfg.testmode = True
        except Exception as e:
            logging.warning(e)
            break

    data["testcase"] = testcase
    return data


def main(args: argparse.Namespace):
    grammar_path = Path(args.grammar_data)
    output_path = Path(args.output)

    with jsonlines.open(grammar_path, "r") as grammar_dataset:
        with jsonlines.open(output_path, "w") as writer:
            writer.write_all(map(
                lambda e: update(e, args.extreme), tqdm(grammar_dataset)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar-data")
    parser.add_argument("--output")
    parser.add_argument("--extreme", action="store_true")
    args = parser.parse_args()

    main(args)
