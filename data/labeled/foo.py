import jsonlines


def foo(obj):
    new_obj = {}
    new_obj['name'] = obj['name']['origin']
    new_obj['description'] = obj['description']['description']
    new_obj['grammar'] = {
        'productions': obj['spec']['grammar'],
        'constraints': obj['spec']['constraints']
    }
    return new_obj


for bar in ['train', 'test']:
    input_dataset = jsonlines.open(f'./{bar}_grammar.jsonl', 'r')
    output_dataset = jsonlines.open(f'./{bar}.jsonl', 'w')
    output_dataset.write_all(map(foo, input_dataset))
