from typing import Generic, List, TypeVar, Union, Sequence
from klinker.typing import GeneralVector
import numpy as np
import torch
import pandas as pd

T = TypeVar("T", np.ndarray, torch.Tensor)

class NamedVector(Generic[T]):
    vectors: T

    def __init__(self, names: List[str], vectors: T):
        self.names = pd.Series(list(range(len(names))), index=names)
        self.vectors = vectors

    def __getitem__(
        self, key: Union[str, int, List[str], List[int], slice]
    ) -> T:
        if isinstance(key, int):
            return self.vectors[key]
        elif isinstance(key, str):
            return self.vectors[self.names.loc[key]]
        elif isinstance(key, Sequence):
            if len(key) == 0:
                return self.vectors[:0]
            elif isinstance(key[0], str):
                return self.vectors[self.names.loc[key]]
            else:
                #TODO fix typing issue
                return self.vectors[key] # type: ignore
        elif isinstance(key, slice):
            return self.vectors[key]
        else:
            raise ValueError(f"Unsupported key type {type(key)} of key {key}")
