import jsonlines
import json

constraints_start_token = '\nConstraints\n\n'
input_start_token = '\nInput\n\n'
input_finish_token = '\nOutput\n\n'

total_count = 0

with jsonlines.open('data/code_contests_train_python_dataset.jsonl') as f:
    for problem in f:
        # if total_count >= 1200: break
        
        if len(problem['solutions']) == 0 or len(problem['incorrect_solutions']) == 0: continue
        
        description = str(problem['description'])
        const_idx = description.find(constraints_start_token)
        start_idx = description.find(input_start_token)
        finish_idx = description.find(input_finish_token)
        # print(description)
        # print(start_idx)
        # print(finish_idx)
        # exit()
        if start_idx >= finish_idx or const_idx >= start_idx: continue
        
        input_constraints = description[const_idx+14:start_idx]
        input_spec =  description[start_idx+8:finish_idx]
        problem['input_constraints'] = input_constraints
        problem['input_spec'] = input_spec
        
        with open('data/code_contests_train_input_spec2.jsonl', 'a', encoding='utf-8') as write_file:
            write_file.write(json.dumps(problem, ensure_ascii=False) + '\n')