import json
import os
from glob import glob
from pathlib import Path
from collections import OrderedDict
from typing import (Any, Optional, cast, )
from tqdm import tqdm

import torch
# import argparse
import jsonlines
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import T5ForConditionalGeneration  # type: ignore [import]

from tokenizer import CountingContextFreeGrammarTokenizer as CCFGTokenizer
from counting_context_free_grammar import CountingContextFreeGrammar as CCFG
from data_loader import MyDataset


def specification_to_grammar(
    model: T5ForConditionalGeneration,
    device: torch.device,
    source_tokenizer: RobertaTokenizer,
    target_tokenizer: CCFGTokenizer,
    specification: str,
    **kwargs
) -> Optional[list[str]]:

    encoding = source_tokenizer.encode(
        specification,
        max_length=512,
        truncation=True,
        add_special_tokens=False
    )

    if len(encoding) == 0:
        return None

    model_inputs = torch.asarray([encoding]).to(device)
    sample_outputs = model.generate(model_inputs, **kwargs)

    for sample_output in sample_outputs:
        grammar = target_tokenizer.decode_to_json(sample_output)
        try:
            CCFG(**grammar)
        except Exception:
            continue
        return grammar
    return None


def label(
    model: T5ForConditionalGeneration,
    device: torch.device,
    source_tokenizer: RobertaTokenizer,
    target_tokenizer: CCFGTokenizer,
    unlabeled_data: list[dict[str, Any]],
    **kwargs,
) -> dict[str, Any]:
    # unlabeled_data = {"name": name, "description": description}
    name = cast(str, unlabeled_data['name'])
    description = cast(str, unlabeled_data['description'])

    specification = MyDataset.get_spec(description)
    grammar = specification_to_grammar(
        model,
        device,
        source_tokenizer,
        target_tokenizer,
        specification,
        **kwargs
    )
    return {
        'name': name,
        'description': description,
        'grammar': grammar,
    }


def main(config: dict[str, Any]) -> None:

    model_dir = Path(config['trainer']['save_dir'])
    checkpoint_paths = glob(str(model_dir / '*'))
    latest_checkpoint_path = max(checkpoint_paths, key=os.path.getctime)

    print(f"Checkpoint: {latest_checkpoint_path}")
    checkpoint = torch.load(latest_checkpoint_path)
    state_dict = checkpoint['model_state_dict']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = T5ForConditionalGeneration.from_pretrained(config['pretrained'])
    model.load_state_dict(state_dict)
    model = model.to(device)

    source_tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])
    target_tokenizer = CCFGTokenizer(source_tokenizer)

    generate_config = config['generate']
    unlabeled_data_path = (
        Path(config['data_dir']) / generate_config['unlabeled_data'])
    unlabeled_data = jsonlines.open(unlabeled_data_path, 'r')
    generate_args = generate_config['args']

    output_path = Path(config['data_dir']) / generate_config['output']
    output = jsonlines.open(output_path, 'w')
    # ouput = {"name": name, "spec": , "grammar": [], constraint: "": []}

    output.write_all(map(
        lambda data: label(
            model, device, source_tokenizer, target_tokenizer, data,
            **generate_args
        ),
        tqdm(unlabeled_data)
    ))


if __name__ == "__main__":
    with open('./config.json') as fp:
        config = json.load(fp, object_hook=OrderedDict)
    main(config)
