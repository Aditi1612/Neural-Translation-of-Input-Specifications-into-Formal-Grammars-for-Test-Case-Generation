"""Check grammar with syntactic error."""

import argparse

import jsonlines
import timeout_decorator

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg
from counting_context_free_grammar import Discriminator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check grammar with syntactic error."
    )
    parser.add_argument("input_file", help="Input file")
    args = parser.parse_args()

    with jsonlines.open(args.input_file) as reader:
        for obj in reader:
            grammar = obj["grammar"]
            productions = grammar["productions"]
            constraints = grammar["constraints"]
            try:

                @timeout_decorator.timeout(10)  # type: ignore
                def foo() -> None:
                    ccfg = Ccfg(productions, constraints)
                    testcase = ccfg.generate(degree=2)

                foo()
            except Exception as e:
                print(grammar)
                print(e)


if __name__ == "__main__":
    main()
