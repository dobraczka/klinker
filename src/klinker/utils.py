import logging
from itertools import chain
from typing import (
    Callable,
    Collection,
    List,
    Literal,
    Sequence,
    TypeVar,
    Union,
    overload,
)

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch import nn

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

# copy-pasted from pykeen
def get_devices(module: nn.Module) -> Collection[torch.device]:
    """Return the device(s) from each components of the model."""
    return {
        tensor.data.device for tensor in chain(module.parameters(), module.buffers())
    }


def get_preferred_device(module: nn.Module) -> torch.device:
    """Return the preferred device."""
    devices = get_devices(module=module)
    if len(devices) == 0:
        raise ValueError(
            "Could not infer device, since there are neither parameters nor buffers."
        )
    if len(devices) == 1:
        return next(iter(devices))
    else:
        raise ValueError("Could not infer device!")


def resolve_device(device: DeviceHint = None) -> torch.device:
    """Resolve a torch.device given a desired device (string)."""
    if device is None or device == "gpu":
        device = "cuda"
    if isinstance(device, str):
        device = torch.device(device)
    if not torch.cuda.is_available() and device.type == "cuda":
        device = torch.device("cpu")
        logger.warning("No cuda devices were available. The model runs on CPU")
    return device


def upgrade_to_sequence(x: Union[X, Sequence[X]]) -> Sequence[X]:
    """Ensure that the input is a sequence.

    .. note ::
        While strings are technically also a sequence, i.e.,

        .. code-block:: python

            isinstance("test", typing.Sequence) is True

        this may lead to unexpected behaviour when calling `upgrade_to_sequence("test")`.
        We thus handle strings as non-sequences. To recover the other behavior, the following may be used:

        .. code-block:: python

            upgrade_to_sequence(tuple("test"))


    :param x: A literal or sequence of literals
    :return: If a literal was given, a one element tuple with it in it. Otherwise, return the given value.

    >>> upgrade_to_sequence(1)
    (1,)
    >>> upgrade_to_sequence((1, 2, 3))
    (1, 2, 3)
    >>> upgrade_to_sequence("test")
    ('test',)
    >>> upgrade_to_sequence(tuple("test"))
    ('t', 'e', 's', 't')
    """
    return x if (isinstance(x, Sequence) and not isinstance(x, str)) else (x,)


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
