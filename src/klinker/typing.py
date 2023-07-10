from typing import Literal, Sequence, Tuple, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import torch

Side = Literal["left", "right"]
GeneralVector = Union[np.ndarray, torch.Tensor]
GeneralVectorLiteral = Literal["np", "pt"]
TorchVectorLiteral: GeneralVectorLiteral = "pt"
NumpyVectorLiteral: GeneralVectorLiteral = "np"
Frame = Union[pd.DataFrame, dd.DataFrame]
