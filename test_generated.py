from discriminator import discriminator
from generator import test_case_generator
import jsonlines
import sys

dataset = 'grammar_test2_res19'

dataset = sys.argv[1]

generator = test_case_generator()
parser = discriminator()
correct = 0
errors = 0
non_error = 0
pass_k = 10

with jsonlines.open(f'data/{dataset}_with_spec.jsonl') as f:
    for idx, problem in enumerate(f, 1):
        test_cases = problem['public_tests']['input']
        test_cases.extend(problem['private_tests']['input'])
        
        is_error = False
        for grammar, const in zip(problem['spec']['generated']['grammar'][:pass_k], problem['spec']['generated']['constraints'][:pass_k]):
            not_passed = False
            for test_case in test_cases:
                try:
                    res = parser(grammer=grammar, constraints=const, test_case=test_case)
                    if not res:
                        not_passed = True
                        break
                except: 
                    is_error = True
                    errors += 1
                    break
            if not (not_passed or is_error):
                correct += 1
                break
print('-----result-----')
print('passed: ', correct)
print('total: ', idx)
print('acc: ', correct/idx)
print('error rate: ', errors / (idx * 10), '\n\n')