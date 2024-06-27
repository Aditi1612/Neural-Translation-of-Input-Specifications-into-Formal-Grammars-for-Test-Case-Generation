import itertools
from typing import Optional

import jsonlines  # type: ignore
from transformers import RobertaTokenizer  # type: ignore [import]

from data_loader import MyDataset
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer

_model_name = "Salesforce/codet5-base"
_source_tokenizer = RobertaTokenizer.from_pretrained(_model_name)
_target_tokenizer = CcfgTokenizer(_source_tokenizer)


def stringified_to_grammar(stringified: str, target_tokenizer: CcfgTokenizer):
    production_encoding, constraint_encoding = (
        target_tokenizer.encode_to_splited(stringified))

    subseparator = target_tokenizer.subseparator
    production_decoding = target_tokenizer.decode(production_encoding)
    constraint_decoding = target_tokenizer.decode(constraint_encoding)

    productions = production_decoding.split(subseparator)
    productions = list(map(str.strip, productions))
    constraints = constraint_decoding.split(subseparator)
    constraints = list(map(str.strip, constraints))

    grammar = {'productions': productions, 'constraints': constraints}
    return grammar


def normalize_grammar(grammar: dict[str, list[str]]) -> dict[str, list[str]]:
    stringified = MyDataset.stringify(grammar)
    grammar = stringified_to_grammar(stringified, _target_tokenizer)
    grammar['constraints'] = normalize_constraints(grammar['constraints'])
    return grammar


def normalize_productions(productions: list[str]) -> list[str]:
    stringified = MyDataset.stringify({
        'productions': productions,
        'constraints': []
    })
    grammar = stringified_to_grammar(stringified, _target_tokenizer)
    return grammar['productions']


def normalize_constraints(constraints: list[str]) -> list[str]:
    constraints = [e for e in constraints if len(e) > 0]
    if len(constraints) == 0:
        return []
    return sorted(["".join(e.split()) for e in constraints])


def get_mode(xs: list[str]):
    groupby_iterable = itertools.groupby(sorted(xs))
    groups = [(k, len(list(v))) for k, v in groupby_iterable]
    groups = sorted(groups, key=lambda e: e[1], reverse=True)
    mode, num_of_mode = groups[0]

    return mode, num_of_mode


def get_filter_list(
    filter_path_1: Optional[str],
    filter_path_2: Optional[str],
    num: int
) -> list[bool]:
    if filter_path_1 is None:
        filter_list_1 = [True] * num
    else:
        filter_list_1 = list(
            all(e) if len(e) > 0 else False
            for e in jsonlines.open(filter_path_1)
        )

    if filter_path_2 is None:
        filter_list_2 = [True] * num
    else:
        filter_list_2 = list(
            all(e) if len(e) > 0 else True
            for e in jsonlines.open(filter_path_2)
        )

    filter_path_3 = "data/generation-result/code-contest/public/test.jsonl"
    filter_list_3 = list(
        all(e) if len(e) > 0 else True
        for e in jsonlines.open(filter_path_3)
    )
    filter_list = [f1 and f2 and f3 for f1, f2, f3 in zip(filter_list_1, filter_list_2, filter_list_3)]
    return filter_list
