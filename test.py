import json
from collections import OrderedDict

import torch
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import T5ForConditionalGeneration  # type: ignore [import]

from data_loader import get_data_loader


def main() -> None:

    with open('./config.json') as fp:
        config = json.load(fp, object_hook=OrderedDict)

    tokenizer = RobertaTokenizer.from_pretrained(config['pretrained'])
    data_loader_args = config['data_loader']['args']
    data_loader_args['shuffle'] = False

    checkpoint = torch.load('./saved/checkpoint-epoch91.pth')
    state_dict = checkpoint['model_state_dict']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = T5ForConditionalGeneration.from_pretrained(config['pretrained'])
    model.load_state_dict(state_dict)
    model = model.to(device)

    data_loader = get_data_loader(
        'data/test_grammar.jsonl', tokenizer, **data_loader_args)

    with torch.no_grad():
        for idx, data in enumerate(data_loader, 1):

            ys = data['target_ids']
            ids = data['source_ids']
            mask = data['source_mask']
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
                print(tokenizer.decode(y, skip_special_tokens=True))

                print("Output:")
                print(output)
                print(tokenizer.decode(output, skip_special_tokens=False))


if __name__ == "__main__":
    main()
