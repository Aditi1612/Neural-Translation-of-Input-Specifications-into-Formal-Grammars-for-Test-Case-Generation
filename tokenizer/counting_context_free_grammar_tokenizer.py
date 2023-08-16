from typing import (cast, Optional, )

import torch
from transformers import BatchEncoding  # type: ignore [import]
from transformers import PreTrainedTokenizerBase  # type: ignore [import]

from ._typing import Tokenizer
from counting_context_free_grammar import CountingContextFreeGrammar as CCFG
from counting_context_free_grammar.counting_context_free_grammar import (
    TokenType,
    SubscriptType,
    Token,
    Terminal,
    Variable,
    Nonterminal,
    DERIVATE_TOKEN,
)


class CountingContextFreeGrammarTokenizer(Tokenizer):
    separator = "\n\n"
    subseparator = "\n"

    def __init__(self, fallback_tokenizer: PreTrainedTokenizerBase) -> None:

        self.nonterminal_table: dict[str, int] = {}
        self.nonterminal_symbol_index = -1
        self.ccfg: Optional[CCFG] = None

        self.fallback_tokenizer = fallback_tokenizer

        self.unk_token_id = self.fallback_tokenizer.unk_token_id
        self.pad_token_id = self.fallback_tokenizer.pad_token_id

        self.nonterminal_token_encoding = (
            self._fallback_encode("symbol"))
        self.variable_token_encoding = (
            self._fallback_encode("variable"))
        self.derivate_token_encoding = (
            self._fallback_encode("to"))

        self.separator_token_encoding = (
            self._fallback_encode("\n"))
        self.subseparator_token_encoding = (
            self._fallback_encode(","))

    def clear(self) -> None:
        self.nonterminal_table = {}
        self.nonterminal_symbol_index = -1
        self.ccfg = None

    def encode(self, text: str, **kwargs) -> list[int]:

        productions_string, constraints_string = text.split(self.separator)
        production_strings = productions_string.split(self.subseparator)
        constraint_strings = constraints_string.split(self.subseparator)

        try:
            self.ccfg = CCFG(production_strings, constraint_strings)
        except Exception as e:
            print(text)
            raise e
        encoding = []
        for word in text.split():
            if word == self.separator:
                encoding.extend(self.separator_token_encoding)
            elif word == self.subseparator:
                encoding.extend(self.subseparator_token_encoding)
            elif word == DERIVATE_TOKEN:
                encoding.extend(self.derivate_token_encoding)
            else:
                encoding.extend(self._encode_token(word))
        self.clear()
        return encoding

    def batch_encode_plus(
        self, batch_text_or_text_pairs: list[str],
        **kwargs,
    ) -> BatchEncoding:
        encodings = [
            torch.tensor(self.encode(text))
            for text in batch_text_or_text_pairs
        ]
        _attention_mask = [
            torch.ones_like(encoding) for encoding in encodings
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
                encodings, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
                _attention_mask, batch_first=True, padding_value=0)

        return BatchEncoding({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })

    def decode(self, token_ids: list[int], **kwargs) -> str:
        decoding: list[str] = []
        for token_id in token_ids:
            decoding.append(self._fallback_decode(token_id))
        return ' '.join(decoding)

    def batch_decode(self, sequences: list[list[int]], **kwargs) -> list[str]:
        return list(map(self.decode, sequences))

    def _get_next_nonterminal_ids(self) -> list[int]:
        self.nonterminal_symbol_index += 1
        index = self.nonterminal_symbol_index
        if index < 4:
            return self._fallback_encode(['X', 'Y', 'Z', 'W'][index])

        return self._fallback_encode(chr(ord('A') + index - 4))

    def _encode_token(self, token: str) -> list[int]:
        self.ccfg = cast(CCFG, self.ccfg)
        token_type = self.ccfg._get_token_type(token)
        try:
            if token_type == TokenType.TERMINAL:
                terminal = cast(Terminal, token)
                return self._encode_terminal(terminal)
            elif token_type == TokenType.NONTERMINAL:
                nonterminal = cast(Nonterminal, token)
                return self._encode_nonterminal(nonterminal)
            elif token_type == TokenType.VARIABLE:
                variable = cast(Variable, token)
                return self._encode_variable(variable)
            else:
                assert False
        except Exception:
            return [self.unk_token_id]

    def _fallback_encode(self, text: str):
        return self.fallback_tokenizer.encode(text, add_special_tokens=False)

    def _fallback_decode(self, text: str):
        return self.fallback_tokenizer.decode(text, skip_special_tokens=True)

    def _encode_terminal(self, terminal: Terminal):
        return self._fallback_encode(terminal)

    def _encode_nonterminal(self, nonterminal: Nonterminal):
        self.ccfg = cast(CCFG, self.ccfg)
        encoding: list[str] = []
        encoding.extend(self.nonterminal_token_encoding)
        fragment, placeholder = self.ccfg._split_token(nonterminal)

        ids = self.nonterminal_table.get(
            fragment, self._get_next_nonterminal_ids())
        encoding.extend(ids)

        if placeholder is not None:
            encoding.extend(self._fallback_encode(placeholder))
        return encoding

    def _encode_variable(self, variable: str):
        self.ccfg = cast(CCFG, self.ccfg)
        encoding: list[str] = []
        encoding.extend(self.variable_token_encoding)
        if self.ccfg._is_counter(variable):
            encoding.extend(self._fallback_encode("counter"))
            variable = variable[1:-1]
        encoding.extend(self._fallback_encode(variable))
        return encoding
