from typing import Generic, List, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
import torch

T = TypeVar("T", np.ndarray, torch.Tensor)


class NamedVector(Generic[T]):
    _vectors: T
    _names: pd.Series

    def __init__(self, names: List[str], vectors: T):
        self.names = names
        self.vectors = vectors

    def _validate(self, names: List[str], vectors: T):
        if not len(set(names)) == len(names):
            raise ValueError("Names must be unique!")
        if len(names) != len(vectors):
            raise ValueError(
                f"Names and vectors must have same length but got len(names)={len(names)} and len(vectors)={len(vectors)}"
            )

    @property
    def names(self) -> List[str]:
        return self._names.index.tolist()

    @names.setter
    def names(self, new_names: List[str]):
        if hasattr(self,"_vectors"):
            self._validate(new_names, self.vectors)
        self._names = pd.Series(list(range(len(new_names))), index=new_names)

    @property
    def vectors(self) -> T:
        return self._vectors

    @vectors.setter
    def vectors(self, new_vectors: T):
        if hasattr(self,"_names"):
            self._validate(self.names, new_vectors)
        self._vectors = new_vectors

    def __getitem__(self, key: Union[str, int, List[str], List[int], slice]) -> T:
        if isinstance(key, int):
            return self.vectors[key]
        elif isinstance(key, str):
            return self.vectors[self._names.loc[key]]
        elif isinstance(key, Sequence):
            if len(key) == 0:
                return self.vectors[:0]
            elif isinstance(key[0], str):
                return self.vectors[self._names.loc[key]]
            else:
                # TODO fix typing issue
                return self.vectors[key]  # type: ignore
        elif isinstance(key, slice):
            return self.vectors[key]
        else:
            raise ValueError(f"Unsupported key type {type(key)} of key {key}")
