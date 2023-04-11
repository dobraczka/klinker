from typing import overload

import numpy as np
import torch

from klinker.typing import (
    GeneralVector,
    GeneralVectorLiteral,
    NumpyVectorLiteral,
    TorchVectorLiteral,
)


@overload
def cast_general_vector(
    vector: GeneralVector, return_type: NumpyVectorLiteral
) -> np.ndarray:
    ...


@overload
def cast_general_vector(
    vector: GeneralVector, return_type: TorchVectorLiteral
) -> torch.Tensor:
    ...


def cast_general_vector(
    vector: GeneralVector,
    return_type: str,
) -> GeneralVector:
    if return_type == TorchVectorLiteral:
        return torch.tensor(vector) if not isinstance(vector, torch.Tensor) else vector
    elif return_type == NumpyVectorLiteral:
        return np.array(vector) if not isinstance(vector, np.ndarray) else vector
    else:
        raise ValueError(f"Unknown return_type: {return_type}!")
