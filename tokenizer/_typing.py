from typing import (Protocol, Union, )

import torch
from transformers import BatchEncoding  # type: ignore [import]


class Tokenizer(Protocol):
    pad_token_id: int

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

    def decode(
        self,
        token_ids: Union[list[int], torch.Tensor],
        **kwargs
    ) -> str:
        pass

    def batch_decode(
        self,
        sequences: Union[list[list[int]], torch.Tensor],
        **kwargs
    ) -> list[str]:
        pass
