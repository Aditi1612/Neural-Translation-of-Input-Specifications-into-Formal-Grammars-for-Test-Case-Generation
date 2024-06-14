import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import torch
import jsonlines
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer
import sacrebleu

from data_loader import MyDataset
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer

# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

def main(config: dict[str, Any]) -> None:
    logging.basicConfig(level=logging.INFO)
    data_dir = Path(config['data_dir'])
    test_data_path = data_dir / config['test_data']
    pretrained_model_name = config['pretrained']
    test_config = config['test']
    model_labeled_data_path = Path(test_config['model_labeled_data'])

    source_tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name)
    target_tokenizer = CcfgTokenizer(source_tokenizer)

    def normalize_list(str_list: list[str]) -> list[str]:
        return sorted(set(str_list))

    def stringified_to_grammar(stringified: str):
        production_encoding, constraint_encoding = target_tokenizer.encode_to_splited(stringified)
        subseparator = target_tokenizer.subseparator
        production_decoding = target_tokenizer.decode(production_encoding)
        constraint_decoding = target_tokenizer.decode(constraint_encoding)

        productions = list(map(str.strip, production_decoding.split(subseparator)))
        constraints = list(map(str.strip, constraint_decoding.split(subseparator)))

        return {'productions': productions, 'constraints': constraints}

    model_labeled_grammar_dict = {}
    with jsonlines.open(model_labeled_data_path) as model_labeled_data:
        for data in model_labeled_data:
            name = data['name']
            model_labeled_grammar_dict[name] = data['grammar']

    bleu_scores = []

    test_dataset = MyDataset(test_data_path)
    for data in tqdm(test_dataset):
        name = data['name']
        model_labeled_grammar = model_labeled_grammar_dict[name]

        model_labeled_grammar_stringified = MyDataset.stringify(model_labeled_grammar)
        model_labeled_grammar = stringified_to_grammar(model_labeled_grammar_stringified)

        model_labeled_productions = normalize_list(model_labeled_grammar['productions'])
        model_labeled_constraints = normalize_list(model_labeled_grammar['constraints'])

        human_labeled_grammar = stringified_to_grammar(data['stringified'])
        human_labeled_productions = normalize_list(human_labeled_grammar["productions"])
        human_labeled_constraints = normalize_list(human_labeled_grammar["constraints"])

        bleu_score_productions = sacrebleu.corpus_bleu(
            [' '.join(human_labeled_productions)],
            [[' '.join(model_labeled_productions)]]
        ).score
        bleu_score_constraints = sacrebleu.corpus_bleu(
            [' '.join(human_labeled_constraints)],
            [[' '.join(model_labeled_constraints)]]
        ).score
        bleu_score_grammar = sacrebleu.corpus_bleu(
            [' '.join(human_labeled_productions + human_labeled_constraints)],
            [[' '.join(model_labeled_productions + model_labeled_constraints)]]
        ).score

        bleu_scores.append((bleu_score_productions, bleu_score_constraints, bleu_score_grammar))

    # Calculate average BLEU scores
    average_bleu_productions = sum(score[0] for score in bleu_scores) / len(bleu_scores)
    average_bleu_constraints = sum(score[1] for score in bleu_scores) / len(bleu_scores)
    average_bleu_grammar = sum(score[2] for score in bleu_scores) / len(bleu_scores)

    logging.info(f"Average BLEU Score Productions: {average_bleu_productions}")
    logging.info(f"Average BLEU Score Constraints: {average_bleu_constraints}")
    logging.info(f"Average BLEU Score Grammar: {average_bleu_grammar}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-labeled-data')
    args = parser.parse_args()

    with open('./config.json') as fp:
        config = json.load(fp)

    defaults = {
        'model_labeled_data': args.model_labeled_data,
    }

    task = 'test'
    task_config = config.setdefault(task, {})
    for k in defaults.keys():
        if getattr(args, k, None) is not None:
            task_config[k] = getattr(args, k)
        task_config.setdefault(k, defaults[k])

    main(config)
