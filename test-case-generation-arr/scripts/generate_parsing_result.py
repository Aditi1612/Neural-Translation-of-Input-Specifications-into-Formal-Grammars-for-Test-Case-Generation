import argparse

import jsonlines  # type: ignore
from tqdm import tqdm  # type: ignore

from generate_generation_result import (
    get_syntactic_validiness_list as get_generalizability_list)


def main(testcase_path: str, grammar_path: str, output_path: str) -> None:
    testcases = [
        e['testcase'] for e in jsonlines.open(testcase_path)]
    grammars = [
        e['grammar'] for e in jsonlines.open(grammar_path)]
    generality_list = [
        get_generalizability_list(testcase, grammar)
        for testcase, grammar in tqdm(zip(testcases, grammars))
    ]
    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(generality_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('grammar')
    parser.add_argument('testcase')
    parser.add_argument('output')
    args = parser.parse_args()

    main(args.testcase, args.grammar, args.output)
