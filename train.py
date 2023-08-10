import json
from collections import OrderedDict
from pathlib import Path

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

    source_tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])
    source_tokenizer.truncation_side = 'left'
    target_tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])

    data_loader_args = config['data_loader']['args']
    data_dir = Path(config['data_dir'])
    train_data_path = data_dir / config['train_data']
    valid_data_path = data_dir / config['valid_data']

    train_data_loader = get_data_loader(
        train_data_path, source_tokenizer, target_tokenizer,
        **data_loader_args
    )

    valid_data_loader = get_data_loader(
        valid_data_path, source_tokenizer, target_tokenizer,
        **data_loader_args
    )

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
        source_tokenizer,
        target_tokenizer,
        valid_data_loader,
        **trainer_args)
    trainer.train()


if __name__ == '__main__':
    main()
