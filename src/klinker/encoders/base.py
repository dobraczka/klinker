from typing import Callable, List, Literal, Optional, Tuple, overload

import numpy as np
import pandas as pd
import torch

from klinker.typing import (
    GeneralVector,
    GeneralVectorLiteral,
    NumpyVectorLiteral,
    TorchVectorLiteral,
)
from klinker.utils import cast_general_vector


class FrameEncoder:
    def validate(self, left: pd.DataFrame, right: pd.DataFrame):
        if len(left.columns) != 1 or len(right.columns) != 1:
            raise ValueError(
                "Input DataFrames must consist of single column containing all attribute values!"
            )

    def prepare(self, left: pd.DataFrame, right: pd.DataFrame):
        pass

    def _encode(
        self, left: pd.DataFrame, right: pd.DataFrame
    ) -> Tuple[GeneralVector, GeneralVector]:
        raise NotImplementedError

    @overload
    def encode(
        self, left: pd.DataFrame, right: pd.DataFrame, return_type: Literal["np"]
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def encode(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        return_type: Literal["pt"],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def encode(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        return_type: GeneralVectorLiteral = "pt",
    ) -> Tuple[GeneralVector, GeneralVector]:
        self.validate(left, right)
        self.prepare(left, right)
        left_enc, right_enc = self._encode(left, right)
        return cast_general_vector(
            left_enc, return_type=return_type
        ), cast_general_vector(right_enc, return_type=return_type)


class TokenizedFrameEncoder(FrameEncoder):
    @property
    def tokenizer_fn(self) -> Callable[[str], List[str]]:
        raise NotImplementedError
