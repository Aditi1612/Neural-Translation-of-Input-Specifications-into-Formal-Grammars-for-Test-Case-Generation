import sys
import jsonlines
import json

dataset_name = sys.argv[1]

dataset = {}
with jsonlines.open('data/test_grammar.jsonl') as f:
    for problem in f:
        idx = problem['name']['index']
        dataset[idx] = problem

print('finish loading')

problems = jsonlines.open(f'outputs/{dataset_name}.jsonl')
output_filename = f'data/{dataset_name}_with_spec.jsonl'
output_file = open(output_filename, 'w', encoding='utf-8')

for problem in problems:
    idx = problem['index']
    name = problem['name']
    generated = problem['generated']
    generated_grammar = []
    generated_const = []

    for g in generated:
        try:
            grammar, const = g.split(' // ')
            grammar = grammar.replace("<R>", "<S>")
            grammar = grammar.replace("<p>", "<s>")
            grammars = grammar.split(' / ')
            consts = const.split(' / ')

        except Exception:
            grammars = g.split(' / ')
            consts = []
            pass

        generated_const.append(consts)
        generated_grammar.append(grammars)

    curr = dataset[idx]
    curr['spec']['generated'] = {
        'grammar': generated_grammar,
        'constraints': generated_const
    }
    output_file.write(json.dumps(curr, ensure_ascii=False) + '\n')

output_file.close()
problems.close()
