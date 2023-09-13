import os
from pathlib import Path

import jsonlines
from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('deepmind/code_contests')

raw_dataset_dir = Path('data/raw')
os.makedirs(raw_dataset_dir, exist_ok=True)

for dataset_type in ['train', 'test', 'valid']:
    path = raw_dataset_dir / f'code_contests_{dataset_type}.jsonl'
    with jsonlines.open(path, 'w') as f:
        f.write_all(tqdm(dataset[dataset_type], desc=f'Writing {path}'))
