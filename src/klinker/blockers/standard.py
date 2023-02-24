from typing import Iterable, List, Optional, Union

import pandas as pd

from klinker.data import KlinkerFrame

from .base import Blocker


class StandardBlocker(Blocker):
    """Fellegi, Ivan P. and Alan B. Sunter. 'A Theory for Record Linkage.' Journal of the American Statistical Association 64 (1969): 1183-1210."""

    def __init__(self, blocking_key: Union[str, List[str]]):
        self.blocking_key = blocking_key

    def _inner_assign(self, kf: KlinkerFrame) -> KlinkerFrame:
        id_col = kf.id_col
        name = kf.name
        blocked = (
            kf[[id_col, self.blocking_key]].groupby(self.blocking_key).agg(list)
        )
        return blocked.rename(columns={id_col: name})

    def _assign(self, tables: Iterable[KlinkerFrame]) -> KlinkerFrame:
        res: Optional[KlinkerFrame] = None
        for tab in tables:
            if res is None:
                res = self._inner_assign(tab)
            else:
                res = res.join(self._inner_assign(tab), how="outer")
        assert res is not None
        return res
