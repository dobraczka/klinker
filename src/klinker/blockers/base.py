import abc
import pandas as pd
from typing import Union, Iterable, List, overload

class Blocker(abc.ABC):

    @abc.abstractmethod
    def assign(self, tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError

