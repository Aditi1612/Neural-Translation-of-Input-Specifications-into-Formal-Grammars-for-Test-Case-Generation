import argparse
import logging
from pathlib import Path

import jsonlines
import timeout_decorator
from tqdm import tqdm

from counting_context_free_grammar import CountingContextFreeGrammar as CCFG


def main(args: argparse.Namespace):
    labeled_path = Path(args.labeled_data)
    output_path = Path(args.output)

    testcase_dataset = []
    with jsonlines.open(labeled_path, "r") as labeled_dataset:
        for data in tqdm(labeled_dataset):
            name = data["name"]
            grammar = data["grammar"]
            productions = grammar["productions"]
            constraints = grammar["constraints"]

            exception = None
            try:
                ccfg = CCFG(productions, constraints)

                @timeout_decorator.timeout(10)
                def generate():
                    return ccfg.generate()

                try:
                    testcase = [generate() for _ in range(10)]
                except timeout_decorator.TimeoutError as e:
                    logging.info(e)
                    ccfg = CCFG(productions, constraints, testmode=True)
                    testcase = [ccfg.generate() for _ in range(10)]
                except Exception as e:
                    if str(e) != "Too many iterations":
                        raise e
                    logging.info(e)
                    ccfg = CCFG(productions, constraints, testmode=True)
                    testcase = [ccfg.generate() for _ in range(10)]

            except Exception as e:
                exception = str(e)
                logging.warning(exception)
                testcase = None

            testcase_data = {
                'name': name,
                'grammar': grammar,
                'testcase': testcase,
            }
            if exception:
                testcase_data['error'] = exception
            testcase_dataset.append(testcase_data)

    with jsonlines.open(output_path, "w") as writer:
        print(output_path)
        writer.write_all(testcase_dataset)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-data")
    parser.add_argument("--output")
    args = parser.parse_args()

    main(args)
