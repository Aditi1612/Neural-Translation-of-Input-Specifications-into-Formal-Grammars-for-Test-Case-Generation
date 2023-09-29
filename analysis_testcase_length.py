import argparse
import json
import logging
import random
from multiprocessing import Pool
from pathlib import Path
from typing import (Any, Optional)

import jsonlines
import torch
import numpy as np
from tqdm import tqdm

from validator import validate_testcase


logger = logging.getLogger(__name__)

# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)  # numpy random seed
random.seed(SEED)  # python random seed


def f(
    i: int,
    data: dict[str, Any],
    config: dict[str, Any]
) -> float:

    solution_prefix = Path(config['solution_prefix'])
    incorrect_solution_prefix = Path(config['incorrect_solution_prefix'])

    validate_testcases_config = config['validate_testcases']
    testcases_type = validate_testcases_config['type']

    if testcases_type == 'fuzzing':
        name = data['name']['origin']
    else:
        name = data['name']

    if testcases_type == 'codecontests_generated':
        testcases = data['generated_tests']['input']
    elif testcases_type == 'codecontests_public':
        testcases = data['public_tests']['input']
    elif testcases_type == 'codecontests_private':
        testcases = data['private_tests']['input']
    elif testcases_type == 'fuzzing':
        testcases = data['fuzzing']['input']
    elif testcases_type == 'model_generated':
        testcases = data['testcase']
    else:
        raise ValueError(f'Unknown testcases type: {testcases_type}')

    if testcases is None or len(testcases) == 0:
        return None

    correct_solution_dir = solution_prefix / name
    incorrect_solution_dir = incorrect_solution_prefix / name

    testcase_lengths = [len(testcase) for testcase in testcases]
    average_testcase_length = np.average(testcase_lengths),
    valid_testcase_lengths = []
    for testcase in testcases:
        is_valid, _ = validate_testcase(
            testcase,
            correct_solution_dir,
            incorrect_solution_dir,
            solution_sampling_seed=42,
            **validate_testcases_config['args']
        )
        if is_valid:
            valid_testcase_lengths.append(len(testcase))

    if len(valid_testcase_lengths) == 0:
        return average_testcase_length, None

    return average_testcase_length, np.average(valid_testcase_lengths)


def main(config: dict[str, Any]):

    # Set variables related to `validate_labeling_testcases`
    validate_testcases_config = config['validate_testcases']
    logging.debug(validate_testcases_config)
    testcases_path = Path(validate_testcases_config['testcase'])

    target_names: Optional[set[str]] = None
    if validate_testcases_config['filter'] is not None:
        filter_testcases_path = Path(validate_testcases_config['filter'])
        with jsonlines.open(filter_testcases_path, 'r') as filter_testcases:
            target_names = {
                data['name'] for data
                in tqdm(filter_testcases, desc='Loading filter testcases')
                if data['testcase'] is not None
            }

    dataset: list[dict[str, Any]] = []
    with jsonlines.open(testcases_path, 'r') as testcases:
        for data in testcases:
            if target_names is None or data['name'] in target_names:
                dataset.append(data)

    with Pool(5) as pool:
        results = list(pool.starmap(
            f, [(i, data, config) for i, data in enumerate(dataset)]))

    filtered_results = list(filter(None, results))

    testcase_lengths = [result[0] for result in filtered_results]
    valid_testcase_lengths = [
        result[1] for result in filtered_results if result[1] is not None
    ]

    print(np.average(testcase_lengths))
    print(np.average(valid_testcase_lengths))


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--testcase')
    parser.add_argument('--type')
    parser.add_argument('--filter')
    args = parser.parse_args()

    with open('./config.json') as fp:
        config = json.load(fp)

    data_dir = Path(config['data_dir'])
    defaults = {
        'testcase': None,
        'type': None,
        'filter': None,
    }

    task = 'validate_testcases'
    task_config = config.setdefault(task, {})
    for k in defaults.keys():
        if getattr(args, k, None) is not None:
            task_config[k] = getattr(args, k)
        task_config.setdefault(k, defaults[k])

    main(config)
