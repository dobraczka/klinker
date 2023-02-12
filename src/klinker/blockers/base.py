import abc
from typing import Iterable, List, Union, overload

import pandas as pd


class Blocker(abc.ABC):
    @abc.abstractmethod
    def assign(self, tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError
