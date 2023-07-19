import jsonlines
import traceback
import sys


from tqdm import tqdm
from counting_context_free_grammar import CountingContextFreeGrammar as CCFG

from discriminator import discriminator as Discriminator
from generator import test_case_generator as TestCaseGenerator

DEFAULT_MAX_ITER = 100


def test_parser(
    parser: Discriminator,
    grammar: list[str],
    constraints: list[str],
    test_cases: list[str]
):
    for test_case in test_cases:
        parsed_result = parser(grammar, constraints, test_case)
        if not parsed_result:
            raise Exception("Wrong parsing result")


def test_generator(
    generator: TestCaseGenerator,
    parser: Discriminator,
    grammar: list[str],
    constraints: list[str],
    k: int = DEFAULT_MAX_ITER
):
    test_cases = [generator(grammar, constraints) for _ in range(k)]
    for test_case in test_cases:
        parsed_result = None
        try:
            parsed_result = parser(grammar, constraints, test_case)
        except Exception:
            pass

        if parsed_result is None:
            raise Exception(f"Wrong generation result\n{test_case}")
    return test_cases


def test_ccfg(
    ccfg: CCFG,
    parser: Discriminator,
    k: int = DEFAULT_MAX_ITER
):
    test_cases = [ccfg.generate() for _ in range(k)]
    for test_case in test_cases:
        parsed_result = None
        try:
            parsed_result = parser(grammar, constraints, test_case)
        except Exception:
            pass

        if parsed_result is None:
            raise Exception(f"Wrong generation result\n{test_case}")


def print_error(name: str, e: Exception):
    print()
    print("#" * 50)
    print(f'{name} exception on idx: {problem_idx}')
    print("Error:", e)
    for k, v in specification.items():
        print(k, v)
    print("#" * 50)
    traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    total = 0

    generator_errors = []
    parser_errors = []
    ccfg_errors = []

    with jsonlines.open('data/train_grammer.jsonl') as problems:
        for problem in tqdm(problems):
            total += 1

            problem_idx = problem['name']['index']
            specification = problem['spec']

            grammar = specification['grammer']
            constraints = specification['constraints']

            parser = Discriminator("test")
            generator = TestCaseGenerator("test")

            public_tests = problem['public_tests']
            private_tests = problem['private_tests']
            test_cases = public_tests['input'] + private_tests['input']

            try:
                test_parser(parser, grammar, constraints, test_cases)
            except Exception:
                parser_errors.append(problem_idx)
                continue

            try:
                test_cases = test_generator(
                    generator, parser, grammar, constraints)
            except Exception as e:
                generator_errors.append(problem_idx)
                print_error("generator", e)

            ccfg = None
            try:
                ccfg = CCFG(grammar, constraints, testmode=True)
                test_ccfg(ccfg, parser, k=DEFAULT_MAX_ITER)
            except Exception as e:
                ccfg_errors.append(problem_idx)
                print_error("CCFG", e)

    parser_passed = total - len(parser_errors)
    generator_passed = total - len(generator_errors) - len(parser_errors)
    ccfg_passed = total - len(ccfg_errors) - len(parser_errors)

    print(f"Parser: {parser_passed}/{total}")
    print(f"Generator: {generator_passed}/{total}")
    print(f"CCFG: {ccfg_passed}/{total}")

    generator_only_error = sorted(set(generator_errors) - set(ccfg_errors))
    ccfg_only_error = sorted(set(ccfg_errors) - set(generator_errors))
    both_error = sorted(set(generator_errors) & set(ccfg_errors))

    print(f"Generator only error {generator_only_error}")
    print(f"CCFG only error {ccfg_only_error}")
    print(f"Both error {both_error}")
