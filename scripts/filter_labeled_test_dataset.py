from pathlib import Path
from tqdm import tqdm


import jsonlines


def main():
    labeled_test_path = Path('data/labeled/test.jsonl')
    unlabeled_train_path = Path(
        'data/unlabeled/code_contests_train_python.jsonl')
    output_path = Path(
        'data/unlabeled/code_contests_train_python_filtered.jsonl')

    labeled_names: set[str] = set()
    with jsonlines.open(labeled_test_path, 'r') as labeled_test:
        for data in tqdm(labeled_test):
            labeled_names.add(data['name'])

    with jsonlines.open(unlabeled_train_path, 'r') as unlabeled_train:
        with jsonlines.open(output_path, 'w') as output:
            output.write_all(
                filter(
                    lambda e: e['name'] not in labeled_names,
                    tqdm(unlabeled_train)
                )
            )


if __name__ == "__main__":
    main()
