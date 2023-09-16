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
    specification: str = "no specification",
    num_testcase_sampling: Optional[int] = None,
) -> bool:
    try:
        rejected_testcase = _test_completeness(
            grammar, testcases, num_testcase_sampling)
        if rejected_testcase is None:
            return True

        logger.warning('Rejected testcase:')
        logger.warning('\n'+rejected_testcase+'\n')
        return False

    except Exception as e:
        logger.info("Parsing Error")
        logger.info(name)
        # logger.warning(specification)
        logger.info(grammar)
        logger.info(e)
        return False


def _test_completeness(
    grammar: dict[str, list[str]],
    testcases: list[str],
    num_testcase_sampling: Optional[int],
) -> Optional[str]:
    discriminator = Discriminator()

    productions = grammar['productions']
    constraints = grammar['constraints']
    if num_testcase_sampling is None:
        sampled_testcases = testcases
    else:
        num_testcase_sampling = min(num_testcase_sampling, len(testcases))
        sampled_testcases = random.sample(testcases, num_testcase_sampling)

    for testcase in sampled_testcases:
        if not discriminator(productions, constraints, testcase):
            return testcase
    return None


def test_soundness(
    grammar: dict[str, list[str]],
    solution_dir: Path,
    *,
    name: str = "no name",
    specification: str = "no specification",
    num_testcase_generation: int,
    num_solution_sampling: Optional[int] = None,
    timeout: float = 2,
) -> bool:
    try:
        ret = _test_soundness(
            grammar,
            solution_dir,
            num_solution_sampling,
            num_testcase_generation,
            timeout,
        )
        if ret is None:
            return True

        generated_testcase, outputs = ret
        logger.warning('Name:')
        logger.warning(name)
        logger.warning('Specification:')
        logger.warning(specification)
        logger.warning('Grammar:')
        logger.warning(grammar)
        logger.warning('Invalid testcase:')
        logger.warning('\n'+generated_testcase+'\n')
        logger.warning('Outputs:')
        for output in outputs:
            logger.warning("#" * 80)
            logger.warning('\n'+output+'\n')
            logger.warning("#" * 80)
        return False

    except Exception as e:
        logger.info("Generation error")
        logger.info(name)
        # logger.warning(specification)
        logger.info(grammar)
        logger.info(e)
        return False


def _test_soundness(
    grammar: dict[str, list[str]],
    solution_dir: Path,
    num_solution_sampling: Optional[int],
    num_testcase_generation: int,
    timeout: float,
) -> Optional[tuple[str, list[str]]]:

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

    if num_solution_sampling is None:
        sampled_solutions = solution_files
    else:
        num_solution_sampling = min(num_solution_sampling, len(solution_files))
        sampled_solutions = random.sample(
            solution_files, num_solution_sampling)

    def normalize_output(output: str) -> str:
        return ' '.join(output.split()).lower()

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
                timeout=timeout,
                text=True
            )
            temp_file.seek(0)
            return process
        completed_processes = map(get_completed_process, sampled_solutions)
        outputs = [ps.stdout for ps in completed_processes]
        outputs = list(map(normalize_output, outputs))

        is_sound = all(output == outputs[0] for output in outputs)
        # is_sound = all(ps.returncode == 0 for ps in completed_processes)
        temp_file.close()

        if not is_sound:
            return generated_testcase, outputs

    return None


def test_correctness(
    grammar: dict[str, list[str]],
    solution_dir: Path,
    testcases: list[str],
    name: str,
    *,
    num_testcase_generation: int,
    num_solution_sampling: Optional[int] = None,
    num_testcase_sampling: Optional[int] = None,
    timeout: float = 2,
) -> bool:
    is_sound = test_soundness(
        grammar, solution_dir,
        name=name,
        num_testcase_generation=num_testcase_generation,
        num_solution_sampling=num_solution_sampling,
        timeout=timeout,
    )

    if not is_sound:
        return False

    is_complete = test_completeness(
        grammar, testcases,
        name=name,
        num_testcase_sampling=num_testcase_sampling
    )

    return is_sound and is_complete
