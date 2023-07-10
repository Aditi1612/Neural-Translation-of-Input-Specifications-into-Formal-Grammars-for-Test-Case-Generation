import jsonlines
import json
import re
import sys

if __name__ == '__main__':
    file_name = sys.argv[1]

    input_spec = {}
    with open(f'data/{file_name}_with_spec.jsonl', 'w', encoding='utf-8') as write_file:
        write_file.write('')

    with jsonlines.open('data/train_input_spec_sample.jsonl') as f:
        for problem in f:
            index = problem['name']['index']
            input_spec[index] = problem
    # print(input_spec.keys())

    # print(input_spec[name].keys())
    with jsonlines.open(f'data/{file_name}.jsonl') as f:
        for idx, problem in enumerate(f, 1):
            # print(idx)
            if '*' in problem['name'] or len(problem['grammer']) == 0 : continue

            # test case도 넣기
            index = problem['name'].split(' - ')[1]
            curr = input_spec[int(index)]
            print(index, curr['name']['name'])

            # sepc = curr[]

            # problem['input_spec'] = problem['description']['spec']
            problem['description'] = curr['description']
            problem['name'] = curr['name']
            problem['public_tests'] = curr['public_tests']
            problem['private_tests'] = curr['private_tests']
            grammer = problem.pop('grammer')
            constraints = problem.pop('constraints')
            spec = curr['description']['spec']
            # origin = curr['description']['origin']
            # description = curr['description']['description']
            # problem['description'] = {'description': description, 'origin': origin}
            problem['spec'] = {'grammer': grammer, 'constraints': constraints, 'spec': spec}
            # print(problem.keys())
            # break


            # problem['description'] = input_spec[name]['description']
            with open(f'data/{file_name}_with_spec.jsonl', 'a', encoding='utf-8') as write_file:
                write_file.write(json.dumps(problem, ensure_ascii=False) + '\n')

