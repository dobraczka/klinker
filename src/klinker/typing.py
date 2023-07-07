from typing import Literal, Sequence, Tuple, Union
import pandas as pd
import dask.dataframe as dd

import numpy as np
import torch

Side = Literal["left","right"]
GeneralVector = Union[np.ndarray, torch.Tensor]
GeneralVectorLiteral = Literal["np", "pt"]
TorchVectorLiteral: GeneralVectorLiteral = "pt"
NumpyVectorLiteral: GeneralVectorLiteral = "np"
Frame = Union[pd.DataFrame, dd.DataFrame]
