import io
import logging
import os
import random
import subprocess
import tempfile
import itertools
from pathlib import Path
from typing import (Optional, Tuple, Any, Union, NamedTuple, cast, )

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg
from counting_context_free_grammar import Discriminator


logger = logging.getLogger(__name__)


TestcasesValidationResult = NamedTuple(
    'TestcasesValidationResult',
    [
        ('num_valid', int),
        ('average_effectiveness', float),
        ('average_effectiveness_without_invalids', float),
    ]
)

TestcaseValidityResult = NamedTuple(
    'TestcaseValidityResult',
    [('validity', float), ('mode', str), ('outputs', list[str])]
)


def is_all_same(xs: list[Any]) -> bool:
    if len(xs) == 0:
        return True
    e = xs[0]
    return all(x == e for x in xs)


def get_stdout(python_file: Path, stdin: io.IOBase) -> Optional[str]:
    stream_position = stdin.tell()
    try:
        process = subprocess.run(
            ['python', python_file],
            capture_output=True,
            stdin=stdin,
            timeout=2,
            text=True
        )
        stdin.seek(stream_position)
        if process.returncode != 0:
            return None
    except Exception:
        return None

    return ' '.join(process.stdout.split()).lower()


def get_validity(
    testcase: Union[str, io.IOBase],
    solution_dir: Path,
    *,
    timeout: float = 2,
    num_solution_sampling: Optional[int] = None
) -> TestcaseValidityResult:

    flag = (type(testcase) == str)

    if flag:
        temp_file = tempfile.TemporaryFile('w+b')
        temp_file.write(testcase.encode('utf-8'))
        temp_file.flush()
    else:
        temp_file = testcase

    position = temp_file.tell()
    temp_file.seek(0)

    solutions = list(solution_dir.iterdir())
    if num_solution_sampling is None:
        num_solution_sampling = len(solutions)
    else:
        num_solution_sampling = min(num_solution_sampling, len(solutions))

    solutions = random.sample(solutions, num_solution_sampling)

    outputs = [get_stdout(solution, temp_file) for solution in solutions]

    def get_mode(xs: list[Any]):
        groupby_iterable = itertools.groupby(sorted(filter(None, outputs)))
        groups = [(k, len(list(v))) for k, v in groupby_iterable]
        groups = sorted(groups, key=lambda e: e[1], reverse=True)
        if len(groups) == 0:
            mode, num_of_mode = '', 0
        mode, num_of_mode = groups[0]

        return mode, num_of_mode

    mode, num_of_mode = get_mode(outputs)
    validity = num_of_mode / len(outputs)

    if validity != 1:
        logger.debug(str(outputs))

    temp_file.seek(position)

    if flag:
        temp_file.close()

    return TestcaseValidityResult(validity, mode, outputs)


def validate_testcase(
    testcase: str,
    correct_solution_dir: Path,
    incorrect_solution_dir: Path,
    *,
    timeout: float = 2,
    num_correct_solution_samples: Optional[int] = None,
    num_incorrect_solution_samples: Optional[int] = None,
    validity_threshold: float = 0.8,
) -> Tuple[bool, int]:

    # correct_solutions = list(correct_solution_dir.iterdir())
    incorrect_solutions = list(incorrect_solution_dir.iterdir())

    # if num_correct_solution_samples is None:
    #     num_correct_solution_samples = len(correct_solutions)
    if num_incorrect_solution_samples is None:
        num_incorrect_solution_samples = len(incorrect_solutions)

    # correct_solutions = random.sample(
    #     correct_solutions, num_correct_solution_samples)
    incorrect_solutions = random.sample(
        incorrect_solutions, num_incorrect_solution_samples)

    with tempfile.TemporaryFile('w+b') as temp_file:
        temp_file.write(testcase.encode('utf-8'))
        temp_file.flush()

        validity, correct_output, _ = get_validity(
            testcase,
            correct_solution_dir,
            timeout=timeout,
            num_solution_sampling=num_correct_solution_samples
        )

        if validity < validity_threshold:
            return False, 0

        incorrect_solution_outputs = [
            get_stdout(solution, temp_file)
            for solution in incorrect_solutions
        ]

    num_distinguished = sum(
        incorrect_output != correct_output
        for incorrect_output in incorrect_solution_outputs
    )
    effectiveness = num_distinguished / len(incorrect_solution_outputs)
    return True, effectiveness


def validate_testcases(
    testcases: list[str],
    correct_solution_dir: Path,
    incorrect_solution_dir: Path,
    *,
    timeout: float = 2,
    num_correct_solution_samples: Optional[int] = None,
    num_incorrect_solution_samples: Optional[int] = None,
) -> TestcasesValidationResult:

    num_valid = 0
    effectivenesses: list[float] = []
    effectivenesses_except_invalids: list[float] = []
    for testcase in testcases:
        is_valid, effectiveness = validate_testcase(
            testcase,
            correct_solution_dir,
            incorrect_solution_dir,
            timeout=timeout,
            num_correct_solution_samples=num_correct_solution_samples,
            num_incorrect_solution_samples=num_incorrect_solution_samples,
        )
        if is_valid:
            num_valid += 1
            effectivenesses_except_invalids.append(effectiveness)
        effectivenesses.append(effectiveness)

    average_effectiveness = sum(effectivenesses) / len(effectivenesses)
    average_effectiveness_without_invalids = (
        sum(effectivenesses) / len(effectivenesses))

    return TestcasesValidationResult(
        num_valid,
        average_effectiveness,
        average_effectiveness_without_invalids
    )


def get_completeness(
    grammar: dict[str, list[str]],
    testcases: list[str],
    *,
    name: str = "no name",
    specification: str = "no specification",
    num_testcase_sampling: Optional[int] = None,
) -> bool:
    try:
        rejected_testcase = _get_completeness(
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


def _get_completeness(
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


def get_soundness(
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
        ret = _get_soundness(
            grammar,
            solution_dir,
            num_solution_sampling,
            num_testcase_generation,
            timeout,
        )
        if ret is None:
            return True

        generated_testcase, _ = ret
        logger.warning('Name:')
        logger.warning(name)
        logger.warning('Specification:')
        logger.warning(specification)
        logger.warning('Grammar:')
        logger.warning(grammar)
        logger.warning('Invalid testcase:')
        logger.warning('\n'+generated_testcase+'\n')
        return False

    except Exception as e:
        logger.info("Generation error")
        logger.info(name)
        # logger.warning(specification)
        logger.info(grammar)
        logger.info(e)
        return False


def _get_soundness(
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

    generated_testcases = (
        [ccfg.generate() for _ in range(num_testcase_generation)])

    for generated_testcase in generated_testcases:
        validity, _, outputs = get_validity(
            generated_testcase,
            solution_dir,
            timeout=timeout,
            num_solution_sampling=num_solution_sampling
        )

        if validity < 0.8:
            return generated_testcase, outputs

    return None


def get_correctness(
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
    is_sound = get_soundness(
        grammar, solution_dir,
        name=name,
        num_testcase_generation=num_testcase_generation,
        num_solution_sampling=num_solution_sampling,
        timeout=timeout,
    )

    if not is_sound:
        return False

    is_complete = get_completeness(
        grammar, testcases,
        name=name,
        num_testcase_sampling=num_testcase_sampling
    )

    return is_sound and is_complete
