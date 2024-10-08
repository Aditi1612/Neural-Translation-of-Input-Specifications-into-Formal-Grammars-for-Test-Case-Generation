import argparse
import json
from multiprocessing import Pool
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, IO

import jsonlines
from utils import CoverageEntry  # type: ignore[import-untyped]
from utils import CoverageResult
from utils import CoverageResultPerTestcase
from utils import get_timeout_dict
from utils import sample_solutions


def append_coverage(
    name: str,
    python_file: Path,
    stdin: IO[bytes],
    timeout: int,
) -> bool:
    """Append coverage data to the coverage file."""
    stream_position = stdin.tell()
    stdin.seek(0)
    flag = False
    try:
        ps = subprocess.run(
            ["coverage", "run", "-a", "--data-file", name, str(python_file)],
            stdin=stdin,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            text=True,
        )
        flag = ps.returncode == 0
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    finally:
        stdin.seek(stream_position)
    return flag


def clean_coverage(name: str) -> bool:
    """Clean the coverage data."""
    ps = subprocess.run(["coverage", "erase", "--data-file", name], check=False)
    return ps.returncode == 0


def get_coverages(name: str) -> list[CoverageEntry]:
    """Get the coverage data."""
    ps = subprocess.run(
        ["coverage", "json", "--data-file", name, "-o", "-"],
        capture_output=True,
        check=False,
    )
    files = json.loads(ps.stdout)["files"]
    coverage_results = [
        CoverageEntry(
            covered_lines=v["summary"]["covered_lines"],
            num_statements=v["summary"]["num_statements"],
        )
        for k, v in files.items()
    ]
    return coverage_results


def run_testcase(
    temp_file: IO[bytes],
    solutions: list[Path],
    timeout: int,
) -> list[CoverageEntry]:
    """Run the testcase with the solutions."""
    try:
        coverage_name = os.path.join(tempfile.mkdtemp(), ".coverage")
        for solution in solutions:
            append_coverage(coverage_name, solution, temp_file, timeout)
        coverage_results = get_coverages(coverage_name)
    finally:
        clean_coverage(coverage_name)
    return coverage_results


# pylint: disable-next=too-many-locals
def f(
    data: dict[str, Any],
    timeout: int,
) -> CoverageResult:
    """Run the coverage analysis for the given testcase."""
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

        correct_coverages = None
        correct_error = None
        try:
            correct_coverages = run_testcase(
                temp_file, correct_solutions, timeout
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            correct_error = str(e)

        incorrect_coverages = None
        incorrect_error = None
        try:
            incorrect_coverages = run_testcase(
                temp_file, incorrect_solutions, timeout
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            incorrect_error = str(e)

        coverage_per_testcase = CoverageResultPerTestcase(
            correct_coverages=correct_coverages,
            incorrect_coverages=incorrect_coverages,
            correct_error=correct_error,
            incorrect_error=incorrect_error,
            correct_solutions=[str(e.name) for e in correct_solutions],
            incorrect_solutions=[str(e.name) for e in incorrect_solutions],
        )
        temp_file.close()
        results.append(coverage_per_testcase)

    coverage_result = CoverageResult(name=name, results=results)
    return coverage_result


def main(testcases_path: Path, output_path: Path):
    """Run the coverage analysis for the given testcases."""
    testcases = jsonlines.open(testcases_path, "r")
    timeout_dict = get_timeout_dict(testcases_path.stem)
    with Pool(4) as pool:
        it = ((e, timeout_dict[e["name"]] * 2) for e in testcases)
        results = pool.starmap(f, it)
        with jsonlines.open(output_path, "w") as writer:
            writer.write_all(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("testcase", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.testcase, args.output)
