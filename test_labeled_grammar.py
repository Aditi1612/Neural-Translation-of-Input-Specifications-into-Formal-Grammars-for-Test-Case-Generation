import json
import logging
from collections import OrderedDict
from pathlib import Path

import jsonlines
import tqdm

from test_grammar import test_completeness
from test_grammar import test_soundness


def main(config):
    execute_config = config['execute']
    data_dir = Path(config['data_dir'])
    solution_dir = Path(execute_config['solution_dir'])

    labeled_path = data_dir / config['train_data']
    labeleds = jsonlines.open(labeled_path, 'r')

    testcases_path = data_dir
    testcases_path /= 'unlabeled'
    testcases_path /= 'code_contests_train_python.jsonl'

    testcases_dict = {}
    with jsonlines.open(testcases_path, 'r') as dataset:
        for data in tqdm.tqdm(dataset, desc='Loading testcases'):
            name = data['name']
            testcases = data['public_tests']['input']
            testcases.extend(data['private_tests']['input'])
            testcases.extend(data['generated_tests']['input'])
            testcases_dict[name] = testcases

    soundness = []
    completeness = []
    for labeled in tqdm.tqdm(labeleds, desc='Testing'):
        grammar = labeled['grammar']
        name = labeled['name']
        testcases = testcases_dict[name]

        is_sound = test_soundness(
            grammar,
            solution_dir / name,
            num_testcases=5,
            num_sampled_solutions=5,
            name=name
        )
        is_complete = test_completeness(grammar, testcases, name=name)

        soundness.append(is_sound)
        completeness.append(is_complete)

    labeleds.close()
    print("Sound:", end=" ")
    print(sum(map(lambda e: 1 if e else 0, soundness)) / len(soundness))
    print("Complete:", end=" ")
    print(sum(map(lambda e: 1 if e else 0, completeness)) / len(completeness))
    print("Sound and Complete:", end=" ")
    print(
        sum(map(
            lambda e: 1 if e[0] and e[1] else 0,
            zip(completeness, soundness)
        ))
        / len(completeness)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with open('./config.json') as fp:
        config = json.load(fp, object_hook=OrderedDict)
    main(config)
