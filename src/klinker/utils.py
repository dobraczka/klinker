import logging
from typing import Callable, List, Literal, TypeVar, overload

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize

from .typing import (
    DeviceHint,
    Frame,
    GeneralVector,
    GeneralVectorLiteral,
    NumpyVectorLiteral,
    TorchVectorLiteral,
)

X = TypeVar("X")

logger = logging.getLogger(__name__)


def resolve_device(device: DeviceHint = None) -> torch.device:
    """Resolve a torch.device given a desired device (string).

    Args:
    ----
      device: DeviceHint:  (Default value = None)

    Returns:
    -------

    """
    # copy-pasted from pykeen
    if device is None or device == "gpu":
        device = "cuda"
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == "cuda":
        device = torch.device("cpu")
        logger.warning("No cuda devices were available. The model runs on CPU")
    return device


def concat_frames(frames: List[Frame]) -> Frame:
    """Concatenate dask or pandas frames.

    Args:
    ----
      frames: List[Frame]: List of dataframes.

    Returns:
    -------
        concatenated dataframes
    """
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
    """Cast a vector to the desired type.

    Args:
    ----
      vector: GeneralVector: Vector to cast
      return_type: GeneralVectorLiteral: Wanted return type.

    Returns:
    -------
        Vector in desired format

    Examples:
    --------
        >>> from klinker.utils import cast_general_vector
        >>> import numpy as np
        >>> arr = np.array([1,2,3])
        >>> cast_general_vector(arr, "pt")
        tensor([1, 2, 3])
        >>> t_arr = cast_general_vector(arr, "pt")
        >>> t_arr
        tensor([1, 2, 3])
        >>> cast_general_vector(t_arr, "np")
        array([1, 2, 3])

    """
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
    """Tokenize rows of series.

    Args:
    ----
      row: pd.Series: row with values to tokenize
      tokenize_fn: Callable[[str], List[str]]: Tokenization function
      min_token_length: int: Discard tokens below this value

    Returns:
    -------
        List of tokens
    """
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
    return list(set(res))
