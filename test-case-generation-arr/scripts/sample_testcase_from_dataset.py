import random
from pathlib import Path

import jsonlines
from tqdm import tqdm


def main():
    raw_data_path = Path('raw-data/code_contests_train.jsonl')
    raw_data = jsonlines.open(raw_data_path, 'r')
    data_dict = {}
    for data in tqdm(raw_data):
        data_dict[data['name']] = data

    ground_truth_grammar_path = Path('data/grammar/ground-truth/test.jsonl')
    names = [e['name'] for e in jsonlines.open(ground_truth_grammar_path, 'r')]

    code_contest = {
        'public': jsonlines.open(
            'data/testcase/code-contest/public/test.jsonl', 'w'),
        'private': jsonlines.open(
            'data/testcase/code-contest/private/test.jsonl', 'w'),
        'generated': jsonlines.open(
            'data/testcase/code-contest/generated/test.jsonl', 'w')
    }

    for name in names:
        data = data_dict[name]
        for k, v in code_contest.items():
            testcases = data[f'{k}_tests']['input']
            num = min(10, len(testcases))
            new_data = {
                'name': name,
                'description': data['description'],
                'testcase': random.sample(testcases, num)
            }
            v.write(new_data)


if __name__ == "__main__":
    main()
