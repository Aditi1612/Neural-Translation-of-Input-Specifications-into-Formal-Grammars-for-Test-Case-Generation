import argparse
import json
import logging
import random
from pathlib import Path
from typing import (Any, )

import jsonlines
import torch
import numpy as np
from tqdm import tqdm
from tqdm.contrib.logging import tqdm_logging_redirect
from data_loader import MyDataset

from validator import validate_testcases


# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)  # numpy random seed
random.seed(SEED)  # python random seed


def main(config: dict[str, Any]):

    solution_prefix = Path(config['solution_prefix'])
    incorrect_solution_prefix = Path(config['incorrect_solution_prefix'])

    # Set variables related to `validate_labeling_testcases`
    validate_testcases_config = config['validate_testcases']
    testcases_path = Path(validate_testcases_config['testcase'])
    is_unlabeled = validate_testcases_config['unlabeled']

    valid_ratios: list[float] = []
    effectivenesses: list[float] = []
    effectivenesses_without_invalids: list[float] = []
    with jsonlines.open(testcases_path, 'r') as testcases:
        for data in tqdm(testcases, desc='Validate testcases'):
            name = data['name']
            if is_unlabeled:
                testcases = data['generated_tests']['input']
            else:
                testcases = data['testcases']['input']
            correct_solution_dir = solution_prefix / name
            incorrect_solution_dir = incorrect_solution_prefix / name

            testcases_validation_result = validate_testcases(
                tqdm(testcases, desc=f'Validate {name}', leave=False),
                correct_solution_dir,
                incorrect_solution_dir,
                **validate_testcases_config['args']
            )
            (
                num_valid, effectiveness, effectiveness_without_invalids
            ) = testcases_validation_result
            valid_ratio = num_valid / len(testcases)
            valid_ratios.append(valid_ratio)
            effectivenesses.append(effectiveness)
            effectivenesses_without_invalids.append(
                effectiveness_without_invalids)

    average_valid_ratio = sum(valid_ratios) / len(valid_ratios)
    average_effectiveness = sum(effectivenesses) / len(effectivenesses)
    average_effectiveness_without_invalids = (
        sum(effectivenesses_without_invalids) /
        len(effectivenesses_without_invalids)
    )
    print(f"Average valid ratio: {average_valid_ratio}")
    print(f"Average effectiveness: {average_effectiveness}")
    print(
        "Average effectiveness without invalids: {}"
        .format(average_effectiveness_without_invalids)
    )


if __name__ == "__main__":
    logger = logging.getLogger('validator')
    logger.addHandler(logging.FileHandler('validate_labeling.log'))

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

    with tqdm_logging_redirect():
        main(config)
