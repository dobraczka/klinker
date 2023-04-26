from typing import Callable, List, Literal, Optional, Tuple, overload

import numpy as np
import pandas as pd
import torch

from ..typing import GeneralVector, GeneralVectorLiteral
from ..utils import cast_general_vector

class FrameEncoder:
    def validate(self, left: pd.DataFrame, right: pd.DataFrame):
        if len(left.columns) != 1 or len(right.columns) != 1:
            raise ValueError(
                "Input DataFrames must consist of single column containing all attribute values!"
            )

    def prepare(self, left: pd.DataFrame, right: pd.DataFrame):
        pass

    def _encode(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        raise NotImplementedError

    @overload
    def encode(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        return_type: Literal["np"],
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def encode(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        return_type: Literal["pt"],
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def encode(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        return_type: GeneralVectorLiteral = "pt",
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> Tuple[GeneralVector, GeneralVector]:
        self.validate(left, right)
        left = left.fillna("")
        right = right.fillna("")
        self.prepare(left, right)
        left_enc, right_enc = self._encode(
            left=left, right=right, left_rel=left_rel, right_rel=right_rel
        )
        return cast_general_vector(
            left_enc, return_type=return_type
        ), cast_general_vector(right_enc, return_type=return_type)


class TokenizedFrameEncoder(FrameEncoder):
    @property
    def tokenizer_fn(self) -> Callable[[str], List[str]]:
        raise NotImplementedError
