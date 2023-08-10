from transformers import BatchEncoding  # type: ignore [import]
from ._typing import Tokenizer


class ContextFreeGrammarTokenizer(Tokenizer):
    def __init__(self) -> None:
        raise NotImplementedError

    def encode(self, text: str, **kwargs) -> list[int]:
        raise NotImplementedError

    def batch_encode_plus(
        self, batch_text_or_text_pairs: list[str],
        **kwargs,
    ) -> BatchEncoding:
        raise NotImplementedError

    def decode(self, token_ids: list[int], **kwargs) -> str:
        raise NotImplementedError

    def batch_decode(self, sequences: list[list[int]], **kwargs) -> list[str]:
        raise NotImplementedError
