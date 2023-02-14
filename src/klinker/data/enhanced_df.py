from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KlinkerFrame:
    df: pd.DataFrame
    name: str
    id_col: str = "id"
