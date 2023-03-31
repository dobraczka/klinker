from typing import Tuple
import pandas as pd
from pykeen.nn.text import TransformerTextEncoder

from klinker.data import KlinkerFrame
from klinker.typing import GeneralVector


class FrameEncoder:
    def encode(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> Tuple[GeneralVector, GeneralVector]:
        raise NotImplementedError


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

    def encode(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> Tuple[GeneralVector, GeneralVector]:
        return self.encoder.encode_all(left.values), self.encoder.encode_all(
            left.values
        )
