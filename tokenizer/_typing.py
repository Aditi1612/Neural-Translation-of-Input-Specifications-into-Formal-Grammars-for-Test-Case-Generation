from typing import (Protocol, )

from transformers import BatchEncoding  # type: ignore [import]


class Tokenizer(Protocol):
    def __init__(self):
        pass

    def encode(self, text: str, **kwargs) -> list[int]:
        pass

    def batch_encode_plus(
        self,
        batch_text_or_text_pairs: list[str],
        **kwargs
    ) -> BatchEncoding:
        pass

    def decode(self, token_ids: list[int], **kwargs) -> str:
        pass

    def batch_decode(self, sequences: list[list[int]], **kwargs) -> list[str]:
        pass
