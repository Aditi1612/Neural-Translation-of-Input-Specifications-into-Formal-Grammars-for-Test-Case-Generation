import jsonlines
from generator import test_case_generator

with jsonlines.open('data/grammer_sample.jsonl') as f:
    for problem in f:
        name = problem['name']
        grammer = problem['grammer']
        const = problem['constraints']

        print(name)
        print()
        generator = test_case_generator()
        res = generator(grammer, const)
        print(res)

        print('\n\n----------')
