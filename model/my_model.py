import copy
from typing import (Optional, Iterable, )

import torch
from transformers import T5ForConditionalGeneration  # type: ignore [import]
from transformers import RobertaTokenizer  # type: ignore [import]
from transformers import GenerationConfig

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg
from tokenizer import CountingContextFreeGrammarTokenizer as CcfgTokenizer


class MyModel(torch.nn.Module):
    def __init__(
        self,
        production_model: T5ForConditionalGeneration,
        constraint_model: T5ForConditionalGeneration,
        source_tokenizer: RobertaTokenizer,
        target_tokenizer: CcfgTokenizer,
    ) -> None:
        super().__init__()
        self.production_model = production_model
        self.constraint_model = constraint_model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        production_labels: Optional[torch.Tensor],
        constraint_labels: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        attention_mask = (input_ids != 0)

        production_output = self.production_model(
                input_ids, attention_mask, labels=production_labels)
        constraint_output = self.constraint_model(
                input_ids, attention_mask, labels=constraint_labels)

        return production_output, constraint_output

    def _generate_decodings(
        self,
        input_ids: torch.Tensor,
        generate_config: GenerationConfig,
        model: T5ForConditionalGeneration
    ) -> list[str]:
        outputs = model.generate(input_ids, generate_config)
        decodings = self.target_tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        return decodings

    def _decoding_to_list(self, decoding: str) -> list[str]:
        subseparator = self.target_tokenizer.subseparator

        iterable: Iterable[str] = decoding.split(subseparator)
        iterable = map(str.strip, iterable)
        iterable = filter(lambda e: len(e) > 0, iterable)
        return list(iterable)

    def _generate_list(
        self,
        input_ids: torch.Tensor,
        generate_config: GenerationConfig,
        model: T5ForConditionalGeneration
    ) -> list[str]:
        decodings = self._generate_decodings(input_ids, generate_config, model)
        decoding = decodings[0]
        return self._decoding_to_list(decoding)

    @staticmethod
    def _post_process_productions(productions: list[str]) -> list[str]:
        processed_productions: list[str] = []
        lhss: set[str] = set()
        for production in productions:
            splited_production = production.split(Ccfg.derivation_token)
            if len(splited_production) < 2:
                continue
            lhs = splited_production[0]
            if lhs in lhss:
                continue
            lhss.add(lhs)
            production = Ccfg.derivation_token.join(splited_production[0:2])
            processed_productions.append(production)
        return processed_productions

    @staticmethod
    def _post_process(
        productions: list[str], constraints: list[str]
    ) -> tuple[list[str], list[str]]:
        productions = copy.deepcopy(productions)
        constraints = copy.deepcopy(constraints)

        def filter_empty_string(string_list: list[str]) -> list[str]:
            return list(filter(lambda e: len(e) > 0, string_list))

        productions = filter_empty_string(productions)
        productions = MyModel._post_process_productions(productions)
        constraints = filter_empty_string(constraints)
        constraints = list(set(constraints))

        return productions, constraints

    def generate(
        self,
        input_ids: torch.Tensor,
        generate_config: GenerationConfig
    ) -> dict[str, list[str]]:

        productions = self._generate_list(
            input_ids, generate_config, self.production_model)
        constraints = self._generate_list(
            input_ids, generate_config, self.constraint_model)
        productions, constraints = (
            MyModel._post_process(productions, constraints))

        return {'productions': productions, 'constraints': constraints}
