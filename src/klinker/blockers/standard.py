from typing import Iterable, List, Optional, Union

import pandas as pd

from .base import Blocker
from klinker.data import KlinkerFrame


class StandardBlocker(Blocker):
    """Fellegi, Ivan P. and Alan B. Sunter. 'A Theory for Record Linkage.' Journal of the American Statistical Association 64 (1969): 1183-1210."""

    def __init__(self, blocking_key: Union[str, List[str]]):
        self.blocking_key = blocking_key

    def _inner_assign(self, kf: KlinkerFrame) -> pd.DataFrame:
        id_col = kf.id_col
        name = kf.name
        blocked = (
            kf.df[[id_col, self.blocking_key]]
            .groupby(self.blocking_key)
            .agg(list)
        )
        return blocked.rename(columns={id_col: name})

    def _assign(self, tables: Iterable[KlinkerFrame]) -> pd.DataFrame:
        res: Optional[pd.DataFrame] = None
        for tab in tables:
            if res is None:
                res = self._inner_assign(tab)
            else:
                res = res.join(self._inner_assign(tab), how="outer")
        return res
