from typing import Callable, List, Literal, overload

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch import nn

from .typing import (
    Frame,
    GeneralVector,
    GeneralVectorLiteral,
    NumpyVectorLiteral,
    TorchVectorLiteral,
)


@overload
def concat_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    ...


@overload
def concat_frames(frames: List[dd.DataFrame]) -> dd.DataFrame:
    ...


def concat_frames(frames: List[Frame]) -> Frame:
    if isinstance(frames[0], pd.DataFrame):
        return pd.concat(frames)
    return dd.concat(frames)


@overload
def cast_general_vector(
    vector: GeneralVector, return_type: Literal["np"]
) -> np.ndarray:
    ...


@overload
def cast_general_vector(
    vector: GeneralVector,
    return_type: Literal["pt"],
) -> torch.Tensor:
    ...


def cast_general_vector(
    vector: GeneralVector,
    return_type: GeneralVectorLiteral,
) -> GeneralVector:
    if return_type == TorchVectorLiteral:
        return torch.tensor(vector) if not isinstance(vector, torch.Tensor) else vector
    elif return_type == NumpyVectorLiteral:
        return np.array(vector) if not isinstance(vector, np.ndarray) else vector
    else:
        raise ValueError(f"Unknown return_type: {return_type}!")


def tokenize_row(
    row: pd.Series,
    tokenize_fn: Callable[[str], List[str]] = word_tokenize,
    min_token_length: int = 1,
) -> List:
    res = []
    for value in row.values:
        res.extend(
            list(
                filter(
                    lambda x: len(x) >= min_token_length,
                    tokenize_fn(str(value)),
                )
            )
        )
    return res
