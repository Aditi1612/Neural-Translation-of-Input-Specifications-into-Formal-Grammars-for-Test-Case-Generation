import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import (Any, )

import jsonlines
from tqdm import tqdm

from grammar_tester import test_completeness
from grammar_tester import test_soundness


def main(config: dict[str, Any]):

    data_dir = Path(config['data_dir'])
    solution_prefix = Path(config['solution_prefix'])

    testcases_path = (
        data_dir / 'unlabeled' / 'code_contests_train_python.jsonl')

    testcases_dictionary: dict[list[str]] = {}
    with jsonlines.open(testcases_path, 'r') as dataset:
        for data in tqdm(dataset, desc='Loading testcases'):
            name = data['name']
            testcases = data['public_tests']['input']
            # testcases.extend(data['private_tests']['input'])
            # testcases.extend(data['generated_tests']['input'])
            testcases_dictionary[name] = testcases

    soundness = []
    completeness = []

    labeled_train_path = data_dir / config['train_data']
    labeled_test_path = data_dir / config['train_data']

    for labeled_path in [labeled_train_path, labeled_test_path]:
        labeleds = jsonlines.open(labeled_path, 'r')

        print(f"Test {labeled_path}")
        for labeled in tqdm(labeleds, desc='Testing'):
            grammar = labeled['grammar']
            name = labeled['name']
            testcases = testcases_dictionary[name]
            solution_dir = solution_prefix / name

            is_sound = test_soundness(
                grammar,
                solution_dir,
                num_testcase_generation=5,
                num_solution_sampling=5,
                name=name
            )
            is_complete = test_completeness(
                grammar, testcases,
                name=name, num_testcase_sampling=10
            )

            soundness.append(is_sound)
            completeness.append(is_complete)

        labeleds.close()
        average_soundness = sum(soundness) / len(soundness)
        average_completeness = sum(completeness) / len(completeness)
        average_correctness = (
            sum(map(lambda e: all(e), zip(completeness, soundness)))
            / len(completeness)
        )
        print(f"Sound: {average_soundness * 100:.2f}%")
        print(f"Complete: {average_completeness * 100:.2f}%")
        print(f"Sound and Complete: {average_correctness * 100:.2f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    with open('./config.json') as fp:
        config = json.load(fp, object_hook=OrderedDict)
    main(config)
