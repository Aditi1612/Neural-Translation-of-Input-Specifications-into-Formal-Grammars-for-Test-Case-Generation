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

from validator import validate_testcases


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
) -> Optional[tuple[float, float, float]]:

    print(i)

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

    if testcases is None or len(testcases) == 0:
        return None

    correct_solution_dir = solution_prefix / name
    incorrect_solution_dir = incorrect_solution_prefix / name

    validation_result = validate_testcases(
        testcases,
        correct_solution_dir,
        incorrect_solution_dir,
        **validate_testcases_config['args']
    )

    num_valid = validation_result.num_valid
    valid_ratio = num_valid / len(testcases)
    effectiveness = validation_result.average_effectiveness
    effectiveness_without_invalids = (
        validation_result.average_effectiveness_without_invalids)

    return valid_ratio, effectiveness, effectiveness_without_invalids


def get_bin_count(xs: list[float], k: int = 10) -> list[int]:
    bin_count = [0 for _ in range(k)]
    for x in xs:
        bin_count[min(int(x * k), k-1)] += 1
    return bin_count


def main(config: dict[str, Any]):

    # Set variables related to `validate_labeling_testcases`
    validate_testcases_config = config['validate_testcases']
    testcases_type = validate_testcases_config['type']
    logging.debug(validate_testcases_config)
    testcases_path = Path(validate_testcases_config['testcase'])

    valid_ratios: list[float] = []
    effectivenesses: list[float] = []
    effectivenesses_without_invalids: list[float] = []

    dataset: list[dict[str, Any]] = []
    with jsonlines.open(testcases_path, 'r') as testcases:
        for data in testcases:
            dataset.append(data)

    with Pool(5) as pool:
        results = list(pool.starmap(
            f, [(i, data, config) for i, data in enumerate(dataset)]))

    num_faileds = sum(result is None for result in results)
    filtered_results = list(filter(None, results))

    valid_ratios = [result[0] for result in filtered_results]
    effectivenesses = [result[1] for result in filtered_results]
    effectivenesses_without_invalids = (
        [result[2] for result in filtered_results])

    average_valid_ratio = sum(valid_ratios) / len(valid_ratios)
    average_effectiveness = sum(effectivenesses) / len(effectivenesses)
    average_effectiveness_without_invalids = (
        sum(effectivenesses_without_invalids)
        / len(effectivenesses_without_invalids)
    )
    failed_ratio = num_faileds / len(results)
    print(
        "{} & Coverage & Valid & Effect. & Effect. w/o Invalids \\\\"
        .format(testcases_type)
    )
    print(f"{(1 - failed_ratio) * 100:.2f}", end=' & ')
    print(f"{average_valid_ratio * 100:.2f}", end=' & ')
    print(f"{average_effectiveness * 100:.2f}", end=' & ')
    print("{:.2f} \\\\".format(average_effectiveness_without_invalids * 100))


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--testcase')
    parser.add_argument('--type')
    args = parser.parse_args()

    with open('./config.json') as fp:
        config = json.load(fp)

    data_dir = Path(config['data_dir'])
    defaults = {
        'testcase': None,
        'type': None,
    }

    task = 'validate_testcases'
    task_config = config.setdefault(task, {})
    for k in defaults.keys():
        if getattr(args, k, None) is not None:
            task_config[k] = getattr(args, k)
        task_config.setdefault(k, defaults[k])

    main(config)
