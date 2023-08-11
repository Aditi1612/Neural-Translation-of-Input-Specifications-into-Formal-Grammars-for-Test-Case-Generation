import json
import os
from glob import glob
from pathlib import Path
from collections import OrderedDict
from typing import (Any, )

import torch
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import T5ForConditionalGeneration  # type: ignore [import]

from data_loader import get_data_loader


def main(config: dict[str, Any]) -> None:

    save_dir = Path(config['trainer']['save_dir'])
    checkpoint_paths = glob(str(save_dir / '*'))
    latest_checkpoint_path = max(checkpoint_paths, key=os.path.getctime)

    print(f"Checkpoint: {latest_checkpoint_path}")
    checkpoint = torch.load(latest_checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = T5ForConditionalGeneration.from_pretrained(config['pretrained'])
    model.load_state_dict(state_dict)
    model = model.to(device)

    data_loader_args = config['data_loader']['args']
    data_loader_args['shuffle'] = False

    source_tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])
    source_tokenizer.truncation_side = 'left'
    target_tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])

    data_loader_args = config['data_loader']['args']
    data_loader_args['shuffle'] = False

    data_dir = Path(config['data_dir'])
    test_data_path = data_dir / config['valid_data']

    data_loader = get_data_loader(
        test_data_path, source_tokenizer, target_tokenizer,
        **data_loader_args
    )

    with torch.no_grad():
        for idx, data in enumerate(data_loader, 1):

            ys = data['targets']['input_ids']
            ids = data['sources']['input_ids']
            mask = data['sources']['attention_mask']

            ys, ids, mask = ys.to(device), ids.to(device), mask.to(device)

            outputs = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=10,
                num_beams=10,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                num_return_sequences=10
            )

            for y, output in zip(ys, outputs):

                print("Goal:")
                target_decoding = (
                    target_tokenizer.decode(y, skip_special_tokens=True))
                print(target_decoding)

                print("Output:")
                output_decoding = (
                    target_tokenizer.decode(output, skip_special_tokens=True))
                print(output_decoding)
                print()


if __name__ == "__main__":
    with open('./config.json') as fp:
        config = json.load(fp, object_hook=OrderedDict)
    main(config)
