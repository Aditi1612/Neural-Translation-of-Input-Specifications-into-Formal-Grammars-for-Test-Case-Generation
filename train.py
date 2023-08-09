import json
from collections import OrderedDict

import torch
import numpy as np
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import T5ForConditionalGeneration  # type: ignore [import]

from data_loader import get_data_loader
from trainer import T5Trainer

device = 'cuda:0'

# Fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)  # pytorch random seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)  # numpy random seed


def main() -> None:

    with open('./config.json') as fp:
        config = json.load(fp, object_hook=OrderedDict)

    tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])

    data_loader_args = config['data_loader']['args']

    train_data_loader = get_data_loader(
        'data/train_grammar.jsonl', tokenizer, **data_loader_args)
    valid_data_loader = get_data_loader(
        'data/test_grammar.jsonl', tokenizer, **data_loader_args)

    model = T5ForConditionalGeneration.from_pretrained(config['pretrained'])
    model = model.to(device)

    optimizer_args = config['optimizer']['args']
    optimizer = torch.optim.Adam(params=model.parameters(), **optimizer_args)
    trainer_args = config['trainer']
    trainer = T5Trainer(
        model,
        optimizer,
        device,
        train_data_loader,
        tokenizer,
        valid_data_loader,
        **trainer_args)
    trainer.train()


if __name__ == '__main__':
    main()
