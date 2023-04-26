from typing import Literal, Sequence, Tuple, Union

import numpy as np
import torch

Side = Literal["left","right"]
GeneralVector = Union[np.ndarray, torch.Tensor]
GeneralVectorLiteral = Literal["np", "pt"]
TorchVectorLiteral: GeneralVectorLiteral = "pt"
NumpyVectorLiteral: GeneralVectorLiteral = "np"
ColumnSpecifier = Union[str, Sequence[str]]
DualColumnSpecifier = Tuple[ColumnSpecifier, ColumnSpecifier]
SingleOrDualColumnSpecifier = Union[str, Sequence[str], DualColumnSpecifier]
