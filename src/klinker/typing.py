from typing import Literal, TypeVar, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch
from class_resolver import Hint

DeviceHint = Hint[torch.device]

Side = Literal["left", "right"]
GeneralVector = Union[np.ndarray, torch.Tensor]
GeneralVectorLiteral = Literal["np", "pt"]
TorchVectorLiteral: GeneralVectorLiteral = "pt"
NumpyVectorLiteral: GeneralVectorLiteral = "np"
Frame = TypeVar("Frame", pd.DataFrame, dd.DataFrame)
SeriesType = Union[pd.Series, dd.Series]
