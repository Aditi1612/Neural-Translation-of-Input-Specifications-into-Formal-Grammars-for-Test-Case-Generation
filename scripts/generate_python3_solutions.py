import os
from enum import IntEnum
from pathlib import Path

from tqdm import tqdm
import jsonlines


class LanguageType(IntEnum):
    UNKNOWN_LANGUAGE = 0
    PYTHON = 1
    CPP = 2
    PYTHON3 = 3
    JAVA = 4


def main():
    dataset_dir = Path('data/unlabeled/')

    os.makedirs(dataset_dir, exist_ok=True)

    for dataset_type in ['train', 'test', 'valid']:
        python_dataset_path = (
            dataset_dir / f'code_contests_{dataset_type}_python.jsonl')

        with jsonlines.open(python_dataset_path, 'r') as python_dataset:
            for data in tqdm(python_dataset):

                name = data['name']
                solutions = data['solutions']['solution']

                solutions_dir = Path(f'data/solutions/{name}')
                os.makedirs(solutions_dir, exist_ok=True)

                for idx, solution in enumerate(solutions):
                    solution_path = solutions_dir / f'{idx}.py'
                    with open(solution_path, 'w') as f:
                        f.write(solution)


if __name__ == "__main__":
    main()
