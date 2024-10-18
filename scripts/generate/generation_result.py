import argparse
from pathlib import Path

import jsonlines
import timeout_decorator  # type: ignore

from counting_context_free_grammar import Discriminator
from utils import GenerationResultPerTestcase
from utils import GenerationResult


def check_syntactic_validness(
    testcase: str,
    grammar: dict[str, list[str]],
) -> GenerationResultPerTestcase:

    @timeout_decorator.timeout(10)
    def _check_syntactic_validness(
        testcase: str,
        grammar: dict[str, list[str]],
    ) -> bool:
        d = Discriminator()
        productions = grammar["productions"]
        constraints = grammar["constraints"]
        return d(productions, constraints, testcase)

    try:
        parsable = _check_syntactic_validness(testcase, grammar)
        return GenerationResultPerTestcase(parsable=parsable, error=None)
    except Exception as e:
        return GenerationResultPerTestcase(parsable=False, error=str(e))


def get_syntactic_validiness_list(
    testcases: list[str],
    grammar: dict[str, list[str]],
) -> list[GenerationResultPerTestcase]:
    validness = [check_syntactic_validness(e, grammar) for e in testcases]
    return validness


def main(
    testcase_path: Path,
    grammar_path: Path,
    output_path: Path,
) -> None:

    testcases_file = jsonlines.open(testcase_path)
    testcases_list = [e.get("testcase", []) for e in testcases_file]

    grammar_objects = jsonlines.open(grammar_path)

    with jsonlines.open(output_path, "w") as writer:
        for grammar_object, testcases in zip(grammar_objects, testcases_list):
            name = grammar_object["name"]
            grammar = grammar_object["grammar"]
            results = get_syntactic_validiness_list(testcases, grammar)
            writer.write(GenerationResult(name=name, results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testcase", type=Path)
    parser.add_argument("--grammar", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    main(args.testcase, args.grammar, args.output)
