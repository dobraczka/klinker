from typing import Dict, Generic, List, Sequence, TypeVar, Union

import numpy as np
import pandas as pd
import torch

from ..typing import GeneralVector

T = TypeVar("T", np.ndarray, torch.Tensor)


def _shorten_tensor_repr(tensor: GeneralVector) -> str:
    t_repr = tensor.__repr__()
    first_bracket_idx = t_repr.find("[")
    last_bracket_idx = t_repr.rfind("]") + 1
    return t_repr[first_bracket_idx:last_bracket_idx]


class NamedVector(Generic[T]):
    _vectors: T
    _names: pd.Series

    def __init__(self, names: Union[List[str], Dict[str, int]], vectors: T):
        if isinstance(names, dict):
            self._names = pd.Series(names)
        elif isinstance(names, pd.Series):
            self._names = names
        else:
            self.names = names
        if not all(isinstance(x, str) for x in self._names.index):
            raise ValueError("The names index must be of type `str`")
        if not (self._names.values == np.arange(len(self._names))).all():
            raise ValueError("Indices must be contiguous!")
        self.vectors = vectors
        self._tensor_lib = np if isinstance(self.vectors, np.ndarray) else torch

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
        if hasattr(self, "_vectors"):
            self._validate(new_names, self.vectors)
        self._names = pd.Series(list(range(len(new_names))), index=new_names)

    @property
    def vectors(self) -> T:
        return self._vectors

    @vectors.setter
    def vectors(self, new_vectors: T):
        if hasattr(self, "_names"):
            self._validate(self.names, new_vectors)
        self._vectors = new_vectors

    @property
    def entity_id_mapping(self) -> Dict[str, int]:
        return self._names.to_dict()

    @property
    def id_entity_mapping(self) -> Dict[int, str]:
        return pd.Series(self._names.index.values, index=self._names).to_dict()

    def _key_handling(
        self, key: Union[str, int, List[str], List[int], slice]
    ) -> Union[int, pd.Series, List[int], slice]:
        if isinstance(key, int):
            return key
        elif isinstance(key, str):
            return self._names.loc[key]
        elif isinstance(key, Sequence):
            if len(key) == 0:
                return slice(0, 0, 1)
            elif isinstance(key[0], str):
                return self._names.loc[key]
            else:
                return key
        elif isinstance(key, slice):
            return key
        else:
            raise ValueError(f"Unsupported key type {type(key)} of key {key}")

    def __getitem__(self, key: Union[str, int, List[str], List[int], slice]) -> T:
        vector_key = self._key_handling(key)
        return self.vectors[vector_key]

    def __setitem__(self, key: Union[str, int, List[str], List[int], slice], value: T):
        if not isinstance(value, np.ndarray) and not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Cannot assign value(s) with type {type(value)}\n To change the names use the `names` variable directly."
            )
        vector_key = self._key_handling(key)
        self.vectors[vector_key] = value

    def __repr__(self) -> str:
        if not hasattr(self, "_vectors"):
            return "NamedVector(names=None, vectors=None)"
        "torch.Tensor" if isinstance(self.vectors, torch.Tensor) else "np.ndarray"
        str_repr = ""
        spacing = "\t    "
        for idx, name_arr_line in enumerate(zip(self.names, self.vectors)):
            name, arr_line = name_arr_line
            line = f'{idx}|"{name}": {_shorten_tensor_repr(arr_line)},\n'
            if str_repr == "":
                str_repr = f"NamedVector({line}"
            else:
                str_repr += f"{spacing}{line}"

        str_repr += f"{spacing}dtype={self.vectors.dtype})"
        return str_repr

    def __len__(self) -> int:
        return len(self.names)

    def __eq__(self, other) -> bool:
        if not isinstance(other, NamedVector):
            return False
        if not self._tensor_lib == other._tensor_lib:
            return False
        if not all(self._names.index == other._names.index):
            return False
        if not all(self._names.values == other._names.values):
            return False
        if not (self._vectors == other._vectors).all():
            return False
        return True

    def concat(self, other: "NamedVector") -> "NamedVector":
        new_vectors = self._tensor_lib.concatenate([self.vectors, other.vectors])
        new_names = self.names + other.names
        return NamedVector(names=new_names, vectors=new_vectors)

    def subset(self, key: Union[str, List[str]]) -> "NamedVector":
        sub_names = self._names.loc[key]
        sub_vectors = self._vectors[sub_names]
        # need to cast to list to ensure contiguous ids
        return NamedVector(names=sub_names.index.tolist(), vectors=sub_vectors)
