import abc
from typing import Iterable, List, Union, overload

import pandas as pd
from klinker.data import KlinkerFrame


class Blocker(abc.ABC):

    def _postprocess(self, blocks: pd.DataFrame) -> pd.DataFrame:
        # remove blocks with only one entry
        max_number_nans = len(blocks.columns) - 1
        return blocks[~(blocks.isnull().sum(axis=1) == max_number_nans)]

    @abc.abstractmethod
    def _assign(self, tables: Iterable[KlinkerFrame]) -> pd.DataFrame:
        raise NotImplementedError

    def assign(self, tables: Iterable[KlinkerFrame]) -> pd.DataFrame:
        return self._postprocess(self._assign(tables))

