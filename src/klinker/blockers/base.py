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
    def _assign(self, tables: Iterable[KlinkerFrame]) -> KlinkerFrame:
        raise NotImplementedError

    def assign(self, tables: Iterable[KlinkerFrame]) -> KlinkerFrame:
        name = "Blocked_" + "_".join(t.name for t in tables)
        res = self._assign(tables)
        res.name = name
        res.id_col = None
        return self._postprocess(res)
