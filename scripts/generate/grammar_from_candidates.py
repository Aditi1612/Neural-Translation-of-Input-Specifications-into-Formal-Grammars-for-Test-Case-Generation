import argparse
import logging
from itertools import product
from pathlib import Path
from typing import (Any, )

import jsonlines  # type: ignore
from timeout_decorator import timeout  # type: ignore
from timeout_decorator import TimeoutError
from tqdm import tqdm  # type: ignore

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg  # type: ignore
from utils import normalize_grammar  # type: ignore


@timeout(10)  # type: ignore
def generate(ccfg: Ccfg) -> str:
    return ccfg.generate(degree=2)  # type: ignore


def update(data: dict[str, Any]) -> dict[str, Any]:
    grammar_candidates = data['grammar_candidates']
    productions = grammar_candidates['productions']
    constraints = grammar_candidates['constraints']
    grammar = None
    for production, constraint in product(productions, constraints):
        if len(production) == 0:
            continue
        try:
            grammar = {'productions': production, 'constraints': constraint}
            grammar = normalize_grammar(grammar)
            assert grammar is not None
            production = grammar['productions']
            constraint = grammar['constraints']
            ccfg = Ccfg(production, constraint)
            generate(ccfg)
            break
        except TimeoutError:
            logging.warning('TimeoutError')
            continue
        except Exception:
            continue
    data['grammar'] = grammar
    del data['grammar_candidates']
    return data


def main(data_path: Path, output_path: Path) -> None:
    with jsonlines.open(output_path, 'w') as writer:
        with jsonlines.open(data_path) as reader:
            grammar_dataset = map(update, reader)
            writer.write_all(tqdm(grammar_dataset, desc='Grammar Generation'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    main(args.data, args.output)
