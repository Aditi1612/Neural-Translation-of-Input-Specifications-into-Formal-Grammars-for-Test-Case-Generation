import jsonlines
import traceback
from typing import (Optional, Callable, )

from tqdm import tqdm
from counting_context_free_grammar import CountingContextFreeGrammar as CCFG

from discriminator import discriminator as Discriminator
from generator import test_case_generator as TestCaseGenerator

DEFAULT_MAX_ITER = 100


def test_parser(
    parse: Callable[[], bool], test_cases: list[str]
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
    parse: Optional[Callable[[], bool]],
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


def print_error(name: str, spec: dict, detail: str) -> None:
    print()
    print("#" * 50)
    print(f'{name} exception on idx: {problem_idx}')
    print("Error:", e)
    for k, v in spec.items():
        print(k, v)
    print(detail)
    print("#" * 50)


if __name__ == "__main__":
    total = 0

    generator_errors = []
    parser_errors = []
    ccfg_errors = []

    with jsonlines.open('data/train_grammer.jsonl') as problems:

        discriminator = Discriminator("test")
        generator = TestCaseGenerator("test")

        for problem in tqdm(problems):
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

            def generate_generator():
                return generator(grammar, constraints)

            def generate_ccfg():
                ccfg = CCFG(grammar, constraints, testmode=True)
                return ccfg.generate()

            # Test Discriminator
            parser_failed = False
            e, detail = test_parser(parse, test_cases)
            if e is not None:
                parser_failed = True
                parser_errors.append(problem_idx)

            # Test Generator
            e, detail = test_generator(
                generate_generator, None if parser_failed else parse)
            if e is not None:
                generator_errors.append(problem_idx)
                # print_error("generator", specification, detail)

            # Test CCFG
            e, detail = test_generator(
                generate_ccfg, None if parser_failed else parse)
            if e is not None:
                ccfg_errors.append(problem_idx)
                print_error("CCFG", specification, detail)

    parser_passed = total - len(parser_errors)
    generator_passed = total - len(generator_errors)
    ccfg_passed = total - len(ccfg_errors)

    print(f"Parser: {parser_passed}/{total}")
    print(f"Generator: {generator_passed}/{total}")
    print(f"CCFG: {ccfg_passed}/{total}")

    generator_only_error = sorted(set(generator_errors) - set(ccfg_errors))
    ccfg_only_error = sorted(set(ccfg_errors) - set(generator_errors))
    both_error = sorted(set(generator_errors) & set(ccfg_errors))

    print(f"Generator only error {generator_only_error}")
    print(f"CCFG only error {ccfg_only_error}")
    print(f"Both error {both_error}")
