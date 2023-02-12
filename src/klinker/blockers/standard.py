from typing import Iterable, List, Optional, Union

import pandas as pd

from .base import Blocker


class StandardBlocker(Blocker):
    """Fellegi, Ivan P. and Alan B. Sunter. 'A Theory for Record Linkage.' Journal of the American Statistical Association 64 (1969): 1183-1210."""

    def __init__(self, blocking_key: Union[str, List[str]]):
        self.blocking_key = blocking_key

    def _inner_assign(self, df: pd.DataFrame) -> pd.DataFrame:
        id_col = df.klinker.id_col
        name = df.klinker.name
        blocked = (
            df.reset_index()[[id_col, self.blocking_key]]
            .groupby(self.blocking_key)
            .agg(list)
        )
        return blocked.rename(columns={id_col: name})

    def assign(self, tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
        res: Optional[pd.DataFrame] = None
        for tab in tables:
            if res is None:
                res = self._inner_assign(tab)
            else:
                res = res.join(self._inner_assign(tab), how="outer")

        assert res is not None  # for mypy
        # remove blocks with only one entry
        max_number_nans = len(res.columns) - 1
        return res[~(res.isnull().sum(axis=1) == max_number_nans)]
