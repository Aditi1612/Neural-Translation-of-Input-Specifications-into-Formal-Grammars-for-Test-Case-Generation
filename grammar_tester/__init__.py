import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import (Optional, cast, )

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg
from counting_context_free_grammar import Discriminator


logger = logging.getLogger(__name__)


def test_completeness(
    grammar: dict[str, list[str]],
    testcases: list[str],
    *,
    name: str = "no name",
    num_testcase_sampling: Optional[int] = None,
) -> bool:
    try:
        return _test_completeness(grammar, testcases, num_testcase_sampling)
    except Exception as e:
        logger.warning(name)
        logger.warning(e)
        return False


def _test_completeness(
    grammar: dict[str, list[str]],
    testcases: list[str],
    num_testcase_sampling: Optional[int],
) -> bool:
    discriminator = Discriminator()

    productions = grammar['productions']
    constraints = grammar['constraints']
    if num_testcase_sampling is None:
        sampled_testcases = testcases
    else:
        num_testcase_sampling = min(num_testcase_sampling, len(testcases))
        sampled_testcases = random.sample(testcases, num_testcase_sampling)

    return all(
        discriminator(productions, constraints, testcase)
        for testcase in sampled_testcases
    )


def test_soundness(
    grammar: dict[str, list[str]],
    solution_dir: Path,
    *,
    name: str = "no name",
    num_testcase_generation: int,
    num_solution_sampling: Optional[int] = None,
    timeout: float = 2,
) -> bool:
    try:
        return _test_soundness(
            grammar,
            solution_dir,
            num_solution_sampling,
            num_testcase_generation,
            timeout,
        )
    except Exception as e:
        logger.info(name)
        logger.info(e)
        return False


def _test_soundness(
    grammar: dict[str, list[str]],
    solution_dir: Path,
    num_solution_sampling: Optional[int],
    num_testcase_generation: int,
    timeout: float,
) -> bool:

    productions = cast(list[str], grammar['productions'])
    constraints = cast(list[str], grammar['constraints'])

    ccfg = Ccfg(productions, constraints, testmode=True)
    if not os.path.exists(solution_dir):
        logger.warning(f'{solution_dir} not exists')

    solution_files = [
        solution_dir / filename
        for filename in os.listdir(solution_dir)
    ]

    generated_testcases = [
        ccfg.generate() for _ in range(num_testcase_generation)
    ]
    flag = True
    for generated_testcase in generated_testcases:

        temp_file = tempfile.TemporaryFile('w+b')
        temp_file.write(generated_testcase.encode('utf-8'))
        temp_file.flush()

        def get_completed_process(
            solution_file: Path
        ) -> subprocess.CompletedProcess:
            temp_file.seek(0)
            process = subprocess.run(
                ['python', solution_file],
                capture_output=True,
                stdin=temp_file,
                timeout=timeout
            )
            temp_file.seek(0)
            return process

        if num_solution_sampling is None:
            sampled_solutions = solution_files
        else:
            num_solution_sampling = (
                min(num_solution_sampling, len(solution_files)))
            sampled_solutions = random.sample(
                solution_files, num_solution_sampling)
        completed_processes = map(get_completed_process, sampled_solutions)

        is_sound = True
        for process, sampled_solution in zip(
                completed_processes, sampled_solutions):
            if process.returncode != 0:
                is_sound = False
                break

        temp_file.close()
        return is_sound

    return flag


def test_correctness(
    grammar: dict[str, list[str]],
    solution_dir: Path,
    testcases: list[str],
    name: str,
    *,
    num_testcase_generation: Optional[int],
    num_solution_sampling: Optional[int] = None,
    num_testcase_sampling: Optional[int] = None,
    timeout: float = 2,
) -> bool:
    is_sound = test_soundness(
        grammar, solution_dir, name,
        num_testcase_generation=num_testcase_generation,
        num_solution_sampling=num_solution_sampling,
        timeout=timeout,
    )
    is_complete = test_completeness(
        grammar, testcases,
        name=name,
        num_testcase_sampling=num_testcase_sampling
    )

    return is_sound and is_complete
