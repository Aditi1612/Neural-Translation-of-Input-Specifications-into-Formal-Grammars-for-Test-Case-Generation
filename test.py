import sys
import jsonlines
from discriminator import discriminator
from generator import test_case_generator

file_name = sys.argv[1]
# file_name = 'train_grammer'
result_directory = "test_results"
g_error_file = f"{result_directory}/result_{file_name}_error_list_g.txt"
p_error_file = f"{result_directory}/result_{file_name}_error_list_p.txt"
g_path_file = f"{result_directory}/result_{file_name}_pass_list_g.txt"
p_path_file = f"{result_directory}/result_{file_name}_pass_list_p.txt"
g_error_reason_file = f"{result_directory}/result_{file_name}_error_reason_g.txt"
p_error_reason_file = f"{result_directory}/result_{file_name}_error_reason_p.txt"

discriminator = discriminator('test')
generator = test_case_generator('test')

with open(p_path_file, 'w', encoding='utf-8') as write_file:
    write_file.write('')     
with open(p_error_file, 'w', encoding='utf-8') as write_file:
    write_file.write('')
with open(p_error_reason_file, 'w', encoding='utf-8') as write_file:
    write_file.write('')

with open(g_path_file, 'w', encoding='utf-8') as write_file:
    write_file.write('')     
with open(g_error_file, 'w', encoding='utf-8') as write_file:
    write_file.write('')
with open(g_error_reason_file, 'w', encoding='utf-8') as write_file:
    write_file.write('')


with jsonlines.open(f'data/{file_name}.jsonl') as f:
    for p_idx, problem in enumerate(f, 1):
        print(p_idx)
        
        # name, idx, _ = problem['name']
        name = problem['name']['name']
        print(name)
        idx = problem['name']['index']
        grammer = problem['spec']['grammer']
        const = problem['spec']['constraints']
        test_cases = problem['public_tests']['input']
        try:
            for test_case in test_cases:
                res = discriminator(grammer, const, test_case)
                if not res: 
                    print(res)
                    break
            else:
                print(res)
            
            with open(f'{p_path_file}', 'a', encoding='utf-8') as write_file:
                write_file.write(f'{name}, {idx}\n')
            
        except Exception as e:
            with open(p_error_file, 'a', encoding='utf-8') as write_file:
                write_file.write(f'{name}, {idx}\n')
                
            with open(p_error_reason_file, 'a', encoding='utf-8') as write_file:
                write_file.write(f'{idx} {name}:\n' + test_case + ('' if test_case[-1] == '\n' else '\n'))
                write_file.write('\t' + str(e) + '\n\n')
                
        try:
            for i in range(3):
                res = generator(grammer, const)
                print(res)
            print()
            
            with open(g_path_file, 'a', encoding='utf-8') as write_file:
                write_file.write(f'{name}, {idx}\n')
            
        except Exception as e:
            print("pass:", i)
            print()
            with open(g_error_file, 'a', encoding='utf-8') as write_file:
                write_file.write(f'{name}, {idx}\n')
                
            with open(g_error_reason_file, 'a', encoding='utf-8') as write_file:
                write_file.write(f'{idx} {name}:\n' + test_case + ('' if test_case[-1] == '\n' else '\n'))
                write_file.write('\t' + str(e) + '\n\n')
                