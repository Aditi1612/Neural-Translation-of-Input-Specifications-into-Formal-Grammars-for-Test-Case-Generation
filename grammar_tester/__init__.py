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


# Fix random seeds for reproducibility
SEED = 42
random.seed(SEED)


def test_completeness(
    grammar: dict[str, list[str]],
    testcases: list[str],
    *,
    name: str = "no name",
    num_sampled_testcases: Optional[int] = None,
) -> bool:
    try:
        return _test_completeness(grammar, testcases, num_sampled_testcases)
    except Exception as e:
        logger.warning(name)
        logger.warning(e)
        return False


def _test_completeness(
    grammar: dict[str, list[str]],
    testcases: list[str],
    num_sampled_testcases: Optional[int],
) -> bool:
    discriminator = Discriminator()

    productions = grammar['productions']
    constraints = grammar['constraints']
    if num_sampled_testcases is None:
        sampled_testcases = testcases
    else:
        sampled_testcases = random.sample(testcases, num_sampled_testcases)

    return all(
        discriminator(productions, constraints, testcase)
        for testcase in sampled_testcases
    )


def test_soundness(
    grammar: dict[str, list[str]],
    solution_dir: Path,
    name: str,
    *,
    num_sampled_solutions: Optional[int] = None,
    num_testcases: Optional[int],
    timeout: float = 2,
) -> bool:
    try:
        return _test_soundness(
            grammar,
            solution_dir / name,
            num_sampled_solutions,
            num_testcases,
            timeout,
        )
    except Exception as e:
        logger.info(name)
        logger.info(e)
        return False


def _test_soundness(
    grammar: dict[str, list[str]],
    solution_dir: Path,
    num_sampled_solutions: Optional[int],
    num_testcases: int,
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

    generated_testcases = [ccfg.generate() for _ in range(num_testcases)]
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

        k: int
        if num_sampled_solutions is None:
            k = len(solution_files)
        else:
            k = min(num_sampled_solutions, len(solution_files))
        sampled_solutions = random.sample(solution_files, k=k)
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
