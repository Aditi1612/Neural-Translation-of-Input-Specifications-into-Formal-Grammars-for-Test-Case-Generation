import argparse
import random
import tempfile
import sys
import subprocess
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional, IO

import jsonlines  # type: ignore


def _has_setrecursionlimit(python_file: Path):
    try:
        with open(python_file, 'r') as f:
            lines = f.readlines()
        if any('setrecursionlimit' in line for line in lines):
            return True
    except Exception:
        pass
    return False


def get_stdout(python_file: Path, stdin: IO, timeout: int) -> Optional[str]:
    stream_position = stdin.tell()
    try:
        process = subprocess.run(
            ['python', str(python_file)],
            capture_output=True,
            stdin=stdin,
            timeout=timeout,
            text=True
        )
        stdin.seek(stream_position)
        if process.returncode != 0:
            return None
    except Exception as e:
        print(e)
        return None

    return ' '.join(process.stdout.split()).lower()


def run_testcase(
    temp_file: IO[bytes],
    solutions: list[Path],
    timeout: int,
) -> list[Optional[str]]:

    position = temp_file.tell()
    temp_file.seek(0)
    outputs = [
        get_stdout(solution, temp_file, timeout)
        for solution in solutions
    ]
    temp_file.seek(position)
    return outputs


def sample_solutions(solution_dir: Path, k: int, seed: int) -> list[Path]:
    random.seed(seed)
    solutions = list(solution_dir.glob('*'))
    k = min(k, len(solutions))
    return random.sample(list(solution_dir.glob('*')), k)


def f(
    i: int,
    data: dict[str, Any],
    correct_solution_prefix: Path,
    incorrect_solution_prefix: Path,
) -> list[dict[str, list[Optional[str]]]]:
    seed = 42

    if i % 100 == 0:
        print(f'Processing {i}th data', file=sys.stderr)

    testcases = data['testcase']
    name = data['name']
    correct_solution_dir = correct_solution_prefix / name
    incorrect_solution_dir = incorrect_solution_prefix / name
    outputs: list[dict[str, list[Optional[str]]]] = []

    correct_solutions = sample_solutions(correct_solution_dir, 10, seed)
    incorrect_solutions = sample_solutions(incorrect_solution_dir, 10, seed)

    for testcase in testcases:
        temp_file = tempfile.TemporaryFile('w+b')
        temp_file.write(testcase.encode('utf-8'))
        temp_file.flush()
        result_dict = {
            'correct': run_testcase(temp_file, correct_solutions, 2),
            'incorrect': run_testcase(temp_file, incorrect_solutions, 2),
        }
        outputs.append(result_dict)
        temp_file.close()
    return outputs


def main(testcases_path: str, output_path: str):
    # Set variables related to `validate_labeling_testcases`
    testcases = jsonlines.open(testcases_path, 'r')
    solution_prefix = Path('./data/solutions/solutions')
    incorrect_solution_prefix = Path('./data/solutions/incorrect_solutions')

    with jsonlines.open(output_path, 'w') as writer:
        with Pool(4) as pool:
            inputs = (
                (i, data, solution_prefix, incorrect_solution_prefix)
                for i, data in enumerate(testcases)
            )
            results = pool.starmap(f, inputs)
            writer.write_all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('testcase')
    parser.add_argument('output')
    args = parser.parse_args()
    main(args.testcase, args.output)
