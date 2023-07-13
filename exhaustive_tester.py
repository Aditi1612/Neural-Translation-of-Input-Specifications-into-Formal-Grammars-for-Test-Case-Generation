import sys

import jsonlines

from discriminator import discriminator
from generator import test_case_generator


def test_parser(parser, grammar, constraints, test_cases):
    for test_case in test_cases:
        parsed_result = parser(grammar, constraints, test_case)
        if not parsed_result:
            raise Exception("Wrong parsing result")


def test_generator(generator, parser, grammar, constraints, k = 10):
    test_cases = [generator(grammar, constraints) for _ in range(k)]
    for test_case in test_cases:
        parsed_result = parser(grammar, constraints, test_case)
        if not parsed_result:
            raise Exception(f"Wrong generation result\n{test_case}")


if __name__ == "__main__":
    total_number = 0
    passed_number = 0

    with jsonlines.open('data/train_grammer.jsonl') as problems:
        for problem in problems:
            total_number += 1

            problem_idx = problem['name']['index']
            specification = problem['spec']

            grammar = specification['grammer']
            constraints = specification['constraints']

            parser = discriminator("test")
            generator = test_case_generator("test")

            public_tests = problem['public_tests']
            private_tests = problem['private_tests']
            test_cases = public_tests['input'] + private_tests['input']

            # Test parser
            parser_failed = True
            try:
                test_parser(parser, grammar, constraints, test_cases)
            except Exception as e:
                print(f'Parser exception on idx: {problem_idx}')
                print(e)
                print(specification)
                print()
                print()
                continue

            try:
                test_generator(generator, parser, grammar, constraints)
            except Exception as e:
                print(f'Generator exception on idx: {problem_idx}')
                print(e)
                print(specification)
                print()
                print()
                continue

            passed_number += 1

    print(f"Result: {passed_number}/{total_number}")
