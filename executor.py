import subprocess
import CountingContextFreeGrammar
from typing import (Optional, Callable, Union, cast, Protocol, )


def excute(problem: dict):
    solution_file = 'solution.py'
    input_file = 'input.txt'

    specification = problem['spec']
    production_strings = cast(list[str], specification['grammar'])
    constraint_strings = cast(list[str], specification['constraints'])
    
    solutions = problem['solutions']

    ccfg = CountingContextFreeGrammar(
        production_strings, constraint_strings, testmode=True)
    
    for _ in range(10):
        generated_test_case = ccfg.generate()
        with open(f'{input_file}', 'w', encoding='utf-8') as f:
            f.write(generated_test_case)

        priv_output = ''

        for solution in solutions:
            with open(f'{solution_file}', 'w', encoding='utf-8') as f:
                f.write(solution)

            res = subprocess.run(f'python {solution_file} < {input_file}', capture_output=True, shell=True)
            output = res.stdout

            if type(output) == bytes:
                output = output.decode('utf-8')

            output = '\n'.join([x.strip() for x in output.split('\n')])

            if not priv_output:
                priv_output = output
                continue

            if priv_output != output:
                return False
            
    return True
            
