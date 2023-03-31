from typing import Sequence, Tuple, Union

import numpy as np
import torch

GeneralVector = Union[np.ndarray, torch.Tensor]
ColumnSpecifier = Union[str, Sequence[str]]
DualColumnSpecifier = Tuple[ColumnSpecifier, ColumnSpecifier]
SingleOrDualColumnSpecifier = Union[str, Sequence[str], DualColumnSpecifier]
