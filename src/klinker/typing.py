from typing import Literal, TypeVar, Union

import numpy as np
import torch
from class_resolver import Hint
import dask.dataframe as dd
import pandas as pd

DeviceHint = Hint[torch.device]

Side = Literal["left", "right"]
GeneralVector = Union[np.ndarray, torch.Tensor]
GeneralVectorLiteral = Literal["np", "pt"]
TorchVectorLiteral: GeneralVectorLiteral = "pt"
NumpyVectorLiteral: GeneralVectorLiteral = "np"

FrameType = TypeVar("FrameType", dd.DataFrame, pd.DataFrame)
SeriesType = TypeVar("SeriesType", pd.Series, dd.Series)
