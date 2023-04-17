from typing import Callable, List, Literal, overload

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize

from .typing import (
    GeneralVector,
    GeneralVectorLiteral,
    NumpyVectorLiteral,
    TorchVectorLiteral,
)


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
