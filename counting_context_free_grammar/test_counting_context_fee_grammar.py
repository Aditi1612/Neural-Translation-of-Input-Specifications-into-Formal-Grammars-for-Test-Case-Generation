import jsonlines
import traceback
from typing import (Optional, Callable, )

from counting_context_free_grammar import CountingContextFreeGrammar as CCFG
from counting_context_free_grammar import InvalidGrammarError
from counting_context_free_grammar import Discriminator

DEFAULT_MAX_ITER = 100


def test_parser(
    parse: Callable[[str], bool], test_cases: list[str]
) -> tuple[Optional[Exception], str]:
    try:
        for test_case in test_cases:
            if not parse(test_case):
                raise Exception("Wrong parsing result")
    except Exception as e:
        detail = traceback.format_exc()
        return e, detail

    return None, ""


def test_generator(
    generate: Callable[[], str],
    parse: Optional[Callable[[str], bool]],
    k: int = DEFAULT_MAX_ITER
) -> tuple[Optional[Exception], str]:

    try:
        test_cases = [generate() for _ in range(k)]
        if parse is None:
            return None, ""

        for test_case in test_cases:
            if not parse(test_case):
                raise Exception(f"Wrong generation result\n{test_case}")

    except Exception as e:
        detail = traceback.format_exc()
        return e, detail

    return None, ""


def print_error(name: str, idx: int, specification: dict, e: Exception):
    print("#" * 50)
    print(f'{name} exception on idx: {idx}')
    print(f"{type(e).__name__}:", e)
    print()
    print("Spec:")
    print(f"Grammar: {specification['grammer']}")
    print(f"Constraints: {specification['constraints']}")
    print("#" * 50)


if __name__ == "__main__":
    total = 0

    grammar_errors = []
    parser_errors = []
    ccfg_errors = []

    with jsonlines.open('data/train_grammer.jsonl') as problems:

        discriminator = Discriminator("test")
        for problem in problems:
            total += 1

            problem_idx = problem['name']['index']
            specification = problem['spec']

            grammar = specification['grammer']
            constraints = specification['constraints']

            public_tests = problem['public_tests']
            private_tests = problem['private_tests']
            test_cases = public_tests['input'] + private_tests['input']

            def parse(testcase):
                discriminator(grammar, constraints, testcase)
                return True

            def generate_ccfg():
                ccfg = CCFG(grammar, constraints, testmode=True)
                return ccfg.generate()

            # Test Discriminator
            parser_failed = False
            e, detail = test_parser(parse, test_cases)
            if e is not None:
                parser_failed = True
                parser_errors.append(problem_idx)

            # Test CCFG
            e, detail = test_generator(
                generate_ccfg, None if parser_failed else parse)
            if isinstance(e, InvalidGrammarError):
                grammar_errors.append((problem_idx, str(e)))

            if e is not None:
                ccfg_errors.append(problem_idx)
                print_error("CCFG", problem_idx, specification, e)

    parser_passed = total - len(parser_errors)
    ccfg_passed = total - len(ccfg_errors)

    print(f"Parser: {parser_passed}/{total}")
    print(f"CCFG: {ccfg_passed}/{total}")
    print(f"Errors: {parser_errors}")
