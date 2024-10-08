import argparse
from pathlib import Path
from typing import Iterable, Optional

import jsonlines
from utils import ExecutionResult  # type: ignore[import-untyped]
from utils import ExecutionResultPerTestcase
from utils import ExecutionSummary
from utils import ExecutionSummaryPerTestcase
from utils import get_mode


def get_answer(execution: ExecutionResultPerTestcase) -> Optional[str]:
    correct_results = [e for e in execution["correct_outputs"] if e is not None]
    if len(correct_results) == 0:
        return None

    answer, _ = get_mode(correct_results)
    return answer


def summarize_execution_per_testcase(
    execution: ExecutionResultPerTestcase,
) -> ExecutionSummaryPerTestcase:
    answer = get_answer(execution)
    correct_result = [e == answer for e in execution["correct_outputs"]]
    incorrect_result = [e == answer for e in execution["incorrect_outputs"]]

    return ExecutionSummaryPerTestcase(
        answer=answer,
        correct_results=correct_result,
        incorrect_results=incorrect_result,
    )


def main(execution_path: Path, output_path: Path) -> None:
    execution_results: Iterable[ExecutionResult]
    execution_results = jsonlines.open(execution_path, "r")

    with jsonlines.open(output_path, "w") as writer:
        for execution_result in execution_results:
            name = execution_result["name"]
            summary_results = []
            for execution_result_per_testcase in execution_result["results"]:
                summary_per_tstcase = summarize_execution_per_testcase(
                    execution_result_per_testcase
                )
                summary_results.append(summary_per_tstcase)
            summary = ExecutionSummary(
                name=name,
                results=summary_results,
            )
            writer.write(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    main(args.execution, args.output)
