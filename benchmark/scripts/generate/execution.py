import argparse
from multiprocessing import Pool
from pathlib import Path
import subprocess
import tempfile
from typing import Any, IO, Optional

import jsonlines
from utils import ExecutionResult  # type: ignore[import-untyped]
from utils import ExecutionResultPerTestcase
from utils import get_timeout_dict
from utils import sample_solutions


def get_stdout(
    python_file: Path, stdin: IO[bytes], timeout: int
) -> Optional[str]:
    stream_position = stdin.tell()
    try:
        process = subprocess.run(
            ["python", str(python_file)],
            capture_output=True,
            stdin=stdin,
            timeout=timeout,
            text=True,
            check=True,
        )
        stdin.seek(stream_position)
        if process.returncode != 0:
            return None
    except Exception as e:  # pylint: disable=broad-except
        print(e)
        return None

    return " ".join(process.stdout.split()).lower()


def run_testcase(
    temp_file: IO[bytes],
    solutions: list[Path],
    timeout: int,
) -> list[Optional[str]]:

    position = temp_file.tell()
    temp_file.seek(0)
    outputs = [
        get_stdout(solution, temp_file, timeout) for solution in solutions
    ]
    temp_file.seek(position)
    return outputs


def f(data: dict[str, Any], timeout: int) -> ExecutionResult:
    seed = 42
    k = 10

    name = data["name"]
    testcases = data.get("testcase", [])

    correct_solutions = sample_solutions(name, "correct", k, seed)
    incorrect_solutions = sample_solutions(name, "incorrect", k, seed)

    results = []
    for testcase in testcases:
        temp_file = tempfile.TemporaryFile("w+b")
        temp_file.write(testcase.encode("utf-8"))
        temp_file.flush()

        correct_outputs = run_testcase(temp_file, correct_solutions, timeout)
        incorrect_outputs = run_testcase(
            temp_file, incorrect_solutions, timeout
        )

        result_per_testcase = ExecutionResultPerTestcase(
            correct_outputs=correct_outputs,
            incorrect_outputs=incorrect_outputs,
            correct_solutions=[str(e.name) for e in correct_solutions],
            incorrect_solutions=[str(e.name) for e in incorrect_solutions],
        )
        temp_file.close()
        results.append(result_per_testcase)

    execution_result = ExecutionResult(name=name, results=results)
    return execution_result


def main(testcases_path: Path, output_path: Path) -> None:
    testcases = jsonlines.open(testcases_path, "r")
    timeout_dict = get_timeout_dict(testcases_path.stem)
    with Pool(4) as pool:
        it = ((e, timeout_dict[e["name"]] * 2) for e in testcases)
        results = pool.starmap(f, it)
        with jsonlines.open(output_path, "w") as writer:
            writer.write_all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testcase", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.testcase, args.output)
