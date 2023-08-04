import sys
import jsonlines
import json

dataset_name = sys.argv[1]

dataset = {}
with jsonlines.open(f'data/test_grammar.jsonl') as f:
    for problem in f:
        idx = problem['name']['index']
        dataset[idx] = problem

print('finish loading')

with jsonlines.open(f'outputs/{dataset_name}.jsonl') as f:
    with open(f'data/{dataset_name}_with_spec.jsonl', 'w', encoding='utf-8') as write_file:
        for problem in f:
            idx = problem['index']
            name = problem['name']
            generated = problem['generated']
            generated_grammar = []
            generated_const = []
            
            for g in generated:
                try:
                    grammar, const = g.split(' // ')
                    grammar = grammar.replace("<R>", "<S>").replace("<p>", "<s>")
                    grammars = grammar.split(' / ')
                    consts = const.split(' / ')
                    
                except:
                    grammars = g.split(' / ')
                    consts = []
                    pass
                    
                generated_const.append(consts)
                generated_grammar.append(grammars)
                
            curr = dataset[idx]
            curr['spec']['generated'] = {'grammar': generated_grammar, 'constraints': generated_const}
            
            write_file.write(json.dumps(curr, ensure_ascii=False) + '\n')