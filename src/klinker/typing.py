from typing import Sequence, Tuple, Union, Literal

import numpy as np
import torch

GeneralVector = Union[np.ndarray, torch.Tensor]
GeneralVectorLiteral = Literal["np","pt"]
TorchVectorLiteral: GeneralVectorLiteral = "pt"
NumpyVectorLiteral: GeneralVectorLiteral = "np"
ColumnSpecifier = Union[str, Sequence[str]]
DualColumnSpecifier = Tuple[ColumnSpecifier, ColumnSpecifier]
SingleOrDualColumnSpecifier = Union[str, Sequence[str], DualColumnSpecifier]
