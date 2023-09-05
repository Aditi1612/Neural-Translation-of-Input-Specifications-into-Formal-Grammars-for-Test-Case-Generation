from typing import (cast, Optional, Union)

import torch
from transformers import BatchEncoding  # type: ignore [import]
from transformers import PreTrainedTokenizerBase  # type: ignore [import]

from ._typing import Tokenizer
from counting_context_free_grammar import CountingContextFreeGrammar as CCFG
from counting_context_free_grammar.counting_context_free_grammar import (
    TokenType,
    Terminal,
    Variable,
    Nonterminal,
    DERIVATE_TOKEN,
    NEW_LINE_TOKEN,
    SPACE_TOKEN,
)


def startswith(a: list, b: list, i: int = 0) -> bool:
    return a[i:i+len(b)] == b


class CountingContextFreeGrammarTokenizer(Tokenizer):
    separator = ";;"
    subseparator = ";"

    def __init__(self, fallback_tokenizer: PreTrainedTokenizerBase) -> None:

        self.nonterminal_table: dict[str, list[int]] = {}
        self.nonterminal_symbol_index = -1
        self.ccfg: Optional[CCFG] = None

        self.fallback_tokenizer = fallback_tokenizer

        self.unk_token_id: int = self.fallback_tokenizer.unk_token_id
        self.pad_token_id: int = self.fallback_tokenizer.pad_token_id

        self.terminal_token_encoding = (
            self._fallback_encode("token"))
        self.nonterminal_token_encoding = (
            self._fallback_encode("symbol"))
        self.variable_token_encoding = (
            self._fallback_encode("variable"))
        self.derivate_token_encoding = (
            self._fallback_encode("to"))

        self.separator_token_encoding = (
            self._fallback_encode(self.separator))
        self.subseparator_token_encoding = (
            self._fallback_encode(self.subseparator))

        self.counter_token_encoding = self._fallback_encode("counter")
        self.newline_token_encoding = self._fallback_encode("newline")
        self.space_token_encoding = self._fallback_encode("blank")

    def encode(self, text: str, **kwargs) -> list[int]:

        productions_string, constraints_string = text.split(self.separator)
        production_strings = productions_string.split(self.subseparator)
        constraint_strings = constraints_string.split(self.subseparator)

        try:
            self.ccfg = CCFG(production_strings, constraint_strings)
        except Exception as e:
            raise e
        encoding = []
        for word in text.split():
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

    def batch_encode_to_splited(
        self,
        batch_text_or_text_pairs: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:

        def split_encoding(encoding: list[int]) -> tuple[list[int], list[int]]:

            index = encoding.index(self.separator_token_encoding[0])
            return (
                encoding[:index],
                encoding[index+len(self.separator_token_encoding):]
            )

        encodings = map(self.encode, batch_text_or_text_pairs)
        splited_encodings = list(map(split_encoding, encodings))
        production_encodings = [
            torch.tensor(e[0], dtype=torch.long)
            for e in splited_encodings
        ]
        constraint_encodings = [
            torch.tensor(e[1], dtype=torch.long)
            for e in splited_encodings
        ]

        production_input_ids = torch.nn.utils.rnn.pad_sequence(
                production_encodings,
                batch_first=True,
                padding_value=self.pad_token_id
        )
        constraint_input_ids = torch.nn.utils.rnn.pad_sequence(
                constraint_encodings,
                batch_first=True,
                padding_value=self.pad_token_id
        )
        return production_input_ids, constraint_input_ids

    def decode(
        self,
        token_ids: Union[list[int], torch.Tensor],
        **kwargs
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        encodings = self._split_encoding(token_ids)
        decodings = [self._decode_token(encoding) for encoding in encodings]

        return ' '.join(decodings)

    def batch_decode(
        self,
        sequences: Union[list[list[int]], torch.Tensor],
        **kwargs
    ) -> list[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return list(map(lambda e: self.decode(e, **kwargs), sequences))

    def _split_encoding(self, token_ids: list[int]) -> list[list[int]]:
        splited_encoding = []
        indexes = []
        split_points: list[list[int]] = [
            self.terminal_token_encoding,
            self.nonterminal_token_encoding,
            self.variable_token_encoding,
            self.derivate_token_encoding,
            self.separator_token_encoding,
            self.subseparator_token_encoding,
            [self.unk_token_id],
        ]

        i = 0
        while i < len(token_ids):
            split_point: list[int]
            flag = False
            for split_point in split_points:
                if startswith(token_ids, split_point, i):
                    flag = True
                    break
            if flag:
                indexes.append(i)
            i += len(split_point) if flag else 1

        for start, end in zip(indexes, indexes[1:] + [len(token_ids)]):
            splited_encoding.append(token_ids[start:end])
        return splited_encoding

    def clear(self) -> None:
        self.nonterminal_table.clear()
        self.nonterminal_symbol_index = -1
        self.ccfg = None

    def decode_to_json(self, encoding: list[int]) -> dict[str, list[str]]:
        output_decoding = self.decode(encoding).strip()
        splited_output = output_decoding.split(self.separator)
        productions = list(filter(
            lambda e: len(e) > 0,
            map(str.strip, splited_output[0].split(self.subseparator))
        ))
        constraints = []
        if len(splited_output) > 1:
            constraints = list(filter(
                lambda e: len(e) > 0,
                map(str.strip,splited_output[1].split(self.subseparator))
            ))
        constraints = list(set(constraints))
        return {
            'productions': productions,
            'constraints': constraints,
        }

    def _decode_token(self, token_ids: list[int]) -> str:
        if startswith(token_ids, self.separator_token_encoding):
            return self.separator
        elif startswith(token_ids, self.subseparator_token_encoding):
            return self.subseparator
        elif startswith(token_ids, self.derivate_token_encoding):
            return DERIVATE_TOKEN
        else:
            try:
                return self._decode_ccfg_token(token_ids)
            except Exception:
                return self._fallback_decode(token_ids)

    def _decode_ccfg_token(self, token_ids: list[int]) -> str:
        if startswith(token_ids, self.terminal_token_encoding):
            return self._decode_terminal(token_ids)
        elif startswith(token_ids, self.nonterminal_token_encoding):
            return self._decode_nonterminal(token_ids)
        elif startswith(token_ids, self.variable_token_encoding):
            return self._decode_variable(token_ids)
        raise Exception("It is not a CCFG token")

    def _decode_terminal(self, token_ids: list[int]) -> str:
        start = len(self.terminal_token_encoding)
        if startswith(token_ids, self.newline_token_encoding, start):
            return NEW_LINE_TOKEN
        elif startswith(token_ids, self.space_token_encoding, start):
            return SPACE_TOKEN
        else:
            return self._fallback_decode(token_ids[start:])

    def _decode_nonterminal(self, token_ids: list[int]) -> str:
        start = len(self.nonterminal_token_encoding)
        fragment = self._fallback_decode(token_ids[start:])
        return f"<{fragment}>"

    def _decode_variable(self, token_ids: list[int]) -> str:
        start = len(self.derivate_token_encoding)
        is_counter = False
        if startswith(token_ids, self.counter_token_encoding, start):
            is_counter = True
            start += len(self.counter_token_encoding)
        fragment = self._fallback_decode(token_ids[start:])
        return f"[{fragment}]" if is_counter else fragment

    def _get_next_nonterminal_ids(self) -> list[int]:
        self.nonterminal_symbol_index += 1
        index = self.nonterminal_symbol_index
        if index < 4:
            return self._fallback_encode(['X', 'Y', 'Z', 'W'][index])

        return self._fallback_encode(chr(ord('A') + index - 4))

    def _encode_ccfg_token(self, token: str) -> list[int]:
        assert self.ccfg is not None

        token_type = self.ccfg._get_token_type(token)
        if token_type == TokenType.TERMINAL:
            terminal = cast(Terminal, token)
            return self._encode_terminal(terminal)
        elif token_type == TokenType.NONTERMINAL:
            nonterminal = cast(Nonterminal, token)
            return self._encode_nonterminal(nonterminal)
        elif token_type == TokenType.VARIABLE:
            variable = cast(Variable, token)
            return self._encode_variable(variable)

    def _encode_token(self, token: str) -> list[int]:
        if token == self.separator:
            return self.separator_token_encoding
        elif token == self.subseparator:
            return self.subseparator_token_encoding
        elif token == DERIVATE_TOKEN:
            return self.derivate_token_encoding
        else:
            try:
                return self._encode_ccfg_token(token)
            except Exception:
                return [self.unk_token_id]

    def _fallback_encode(self, text: str) -> list[int]:
        return self.fallback_tokenizer.encode(
            text, add_special_tokens=False)

    def _fallback_decode(self, token_ids: list[int]) -> str:
        return self.fallback_tokenizer.decode(
            token_ids, skip_special_tokens=True)

    def _encode_terminal(self, terminal: Terminal) -> list[int]:
        encoding: list[int] = []
        encoding.extend(self.terminal_token_encoding)

        if terminal == NEW_LINE_TOKEN:
            encoding.extend(self.newline_token_encoding)
        elif terminal == SPACE_TOKEN:
            encoding.extend(self.space_token_encoding)
        else:
            encoding.extend(self._fallback_encode(terminal))

        return encoding

    def _encode_nonterminal(self, nonterminal: Nonterminal) -> list[int]:
        self.ccfg = cast(CCFG, self.ccfg)
        encoding: list[int] = []
        encoding.extend(self.nonterminal_token_encoding)
        fragment, placeholder = self.ccfg._split_token(nonterminal)

        if fragment not in self.nonterminal_table:
            self.nonterminal_table[fragment] = self._get_next_nonterminal_ids()

        ids = self.nonterminal_table[fragment]
        encoding.extend(ids)

        if placeholder is not None:
            encoding.extend(self._fallback_encode('_' + placeholder))
        return encoding

    def _encode_variable(self, variable: Variable) -> list[int]:
        self.ccfg = cast(CCFG, self.ccfg)
        encoding: list[int] = []
        encoding.extend(self.variable_token_encoding)
        if self.ccfg._is_counter(variable):
            encoding.extend(self.counter_token_encoding)
            variable = Variable(variable[1:-1])
        encoding.extend(self._fallback_encode(variable))
        return encoding
