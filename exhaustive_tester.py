import jsonlines
import traceback
import sys
import logging


from tqdm import tqdm
from counting_context_free_grammar import CountingContextFreeGrammar as CCFG

from discriminator import Discriminator
from generator import TestCaseGenerator


DEFAULT_MAX_ITER = 1


def test_parser(parser, grammar, constraints, test_cases):
    for test_case in test_cases:
        parsed_result = parser(grammar, constraints, test_case)
        if not parsed_result:
            raise Exception("Wrong parsing result")


def test_generator(generator, parser, grammar, constraints, k=DEFAULT_MAX_ITER):
    test_cases = [generator(grammar, constraints) for _ in range(k)]
    for test_case in test_cases:
        parsed_result = parser(grammar, constraints, test_case)
        if not parsed_result:
            raise Exception(f"Wrong generation result\n{test_case}")
    return test_cases


def test_ccfg(ccfg, parser, k=DEFAULT_MAX_ITER):
    for _ in range(k):
        test_case = ccfg.generate()
        try:
            parser(grammar, constraints, test_case)
        except Exception as e:
            print(test_case)
            raise e


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
            except Exception:
                generator_errors.append(problem_idx)
                continue

            ccfg = None
            try:
                ccfg = CCFG(grammar, constraints)
                test_ccfg(ccfg, parser, k=DEFAULT_MAX_ITER)
            except Exception as e:
                ccfg_errors.append(problem_idx)
                print()
                print("#" * 50)
                print(f'CCFG exception on idx: {problem_idx}')
                print("Error:", e)
                for k, v in specification.items():
                    print(k, v)
                print(ccfg)
                print("#" * 50)
                traceback.print_exc(file=sys.stdout)

    parser_passed = total - len(parser_errors)
    generator_passed = total - len(generator_errors)
    ccfg_passed = total - len(ccfg_errors)

    print(f"Parser: {parser_passed}/{total}")
    print(f"Generator: {generator_passed}/{total}")
    print(f"CCFG: {ccfg_passed}/{total}")

    print(ccfg_errors)
