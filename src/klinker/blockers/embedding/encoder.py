from typing import Tuple

import pandas as pd
from pykeen.nn.text import TransformerTextEncoder

from klinker.typing import GeneralVector


class FrameEncoder:
    def validate(self, left: pd.DataFrame, right: pd.DataFrame):
        if len(left.columns) != 1 or len(right.columns) != 1:
            raise ValueError(
                "Input DataFrames must consist of single column containing all attribute values!"
            )

    def _encode(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> Tuple[GeneralVector, GeneralVector]:
        raise NotImplementedError

    def encode(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> Tuple[GeneralVector, GeneralVector]:
        self.validate(left, right)
        return self._encode(left, right)


class TransformerTextFrameEncoder(FrameEncoder):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-base-cased",
        max_length: int = 512,
    ):
        self.encoder = TransformerTextEncoder(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            max_length=max_length,
        )

    def _encode(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> Tuple[GeneralVector, GeneralVector]:
        return self.encoder.encode_all(left.values), self.encoder.encode_all(
            left.values
        )
