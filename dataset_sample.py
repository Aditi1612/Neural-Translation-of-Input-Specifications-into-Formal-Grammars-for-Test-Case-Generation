import jsonlines
import json


problem_list = []    
with jsonlines.open('data/code_contests_train_input_spec2.jsonl') as f:
    for problem in f:
        # print(problem['input_spec'][8:])
        problem['input_spec'] = problem['input_spec']
            # exit()
        problem_list.append(problem)
        
problem_list.sort(key= lambda x: ((len(x['solutions']) + len(x['incorrect_solutions']))//2, len(x['solutions'])))
problem_list.reverse()

with open('data/train_input_spec_sample3.jsonl', 'a', encoding='utf-8') as write_file:
    for problem in problem_list[:1200]:
        write_file.write(json.dumps(problem, ensure_ascii=False) + '\n')