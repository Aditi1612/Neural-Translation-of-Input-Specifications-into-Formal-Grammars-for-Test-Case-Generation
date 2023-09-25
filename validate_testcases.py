import argparse
import json
import logging
import random
from multiprocessing import Pool
from pathlib import Path
from typing import (Any, )

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
) -> tuple[float, float, float]:

    print(i)

    solution_prefix = Path(config['solution_prefix'])
    incorrect_solution_prefix = Path(config['incorrect_solution_prefix'])

    validate_testcases_config = config['validate_testcases']
    is_unlabeled = validate_testcases_config['unlabeled']

    name = data['name']
    if is_unlabeled:
        testcases = data['generated_tests']['input']
    else:
        testcases = data['testcases']['input']
    correct_solution_dir = solution_prefix / name
    incorrect_solution_dir = incorrect_solution_prefix / name

    validation_result = validate_testcases(
        testcases,
        correct_solution_dir,
        incorrect_solution_dir,
        **validate_testcases_config['args']
    )

    num_valid = validation_result.num_valid
    effectiveness = validation_result.average_effectiveness
    effectiveness_without_invalids = (
        validation_result.average_effectiveness_without_invalids)

    return num_valid, effectiveness, effectiveness_without_invalids


def main(config: dict[str, Any]):

    # Set variables related to `validate_labeling_testcases`
    validate_testcases_config = config['validate_testcases']
    testcases_path = Path(validate_testcases_config['testcase'])

    valid_ratios: list[float] = []
    effectivenesses: list[float] = []
    effectivenesses_without_invalids: list[float] = []

    dataset: list[dict[str, Any]] = []
    with jsonlines.open(testcases_path, 'r') as testcases:
        for data in testcases:
            dataset.append(data)

    with Pool(5) as pool:
        results = pool.starmap(
            f,
            [(i, data, config) for i, data in enumerate(dataset)]
        )
        valid_ratios = [result[0] for result in results]
        effectivenesses = [result[1] for result in results]
        effectivenesses_without_invalids = [result[2] for result in results]

    average_valid_ratio = sum(valid_ratios) / len(valid_ratios)
    average_effectiveness = sum(effectivenesses) / len(effectivenesses)
    average_effectiveness_without_invalids = (
        sum(effectivenesses_without_invalids)
        / len(effectivenesses_without_invalids)
    )
    print(f"Average valid ratio: {average_valid_ratio}")
    print(f"Average effectiveness: {average_effectiveness}")
    print(
        "Average effectiveness without invalids: {}"
        .format(average_effectiveness_without_invalids)
    )


if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--testcase')
    parser.add_argument('--unlabeled', action='store_true')
    args = parser.parse_args()

    with open('./config.json') as fp:
        config = json.load(fp)

    data_dir = Path(config['data_dir'])
    defaults = {
        'testcase': None,
        'unlabeled': False,
    }

    task = 'validate_testcases'
    task_config = config.setdefault(task, {})
    for k in defaults.keys():
        if getattr(args, k, None) is not None:
            task_config[k] = getattr(args, k)
        task_config.setdefault(k, defaults[k])

    main(config)
