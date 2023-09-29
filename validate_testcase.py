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
) -> Optional[tuple[float, Optional[float], Optional[float]]]:
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
        testcases = data['testcase'][:10]
        testcases = testcases[:min(len(testcases), 10)]

    if testcases is None or len(testcases) == 0:
        return None

    correct_solution_dir = solution_prefix / name
    incorrect_solution_dir = incorrect_solution_prefix / name

    validation_result = validate_testcases(
        testcases,
        correct_solution_dir,
        incorrect_solution_dir,
        solution_sampling_seed=42,
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
    coverage = len(filtered_results) / len(results)
    valid_ratios = [e[0] for e in filtered_results]
    average_valid_ratio = sum(valid_ratios) / len(valid_ratios)
    effectivenesses = [e[1] for e in filtered_results]
    average_effectiveness = sum(effectivenesses) / len(effectivenesses)

    effectivenesses_without_invalids = (
        list(filter(None, [e[2] for e in filtered_results])))
    coverage_without_invalids = (
        len(effectivenesses_without_invalids) / len(results))
    average_effectiveness_without_invalids = (
        sum(effectivenesses_without_invalids)
        / len(effectivenesses_without_invalids)
    )

    print(
        "{} & Coverage & Valid & InEffectiveness & Coverage & InEffectiveness \\\\"
        .format(testcases_type)
    )
    # Coverage
    print(f"{coverage * 100:.2f}", end=' & ')
    # Valid
    print("{:.2f} {{\\small($\\pm{:.2f}$)}}".format(
        average_valid_ratio * 100, np.std(valid_ratios) * 100
    ), end=' & ')
    # Effectiveness
    print("{:.2f} {{\\small($\\pm${:.2f})}} &".format(
        (1 - average_effectiveness) * 100, np.std(effectivenesses) * 100
    ))

    # Coverage w/o invalids
    print(f"{coverage_without_invalids * 100:.2f} ", end=' & ')
    # Effectiveness w/o invalids
    print("{:.2f} {{\\small($\\pm{:.2f}$)}} \\\\".format(
        (1 - average_effectiveness_without_invalids) * 100,
        np.std(effectivenesses_without_invalids) * 100,
    ))


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
