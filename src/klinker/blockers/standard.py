from typing import Iterable, List, Optional, Union

import pandas as pd

from .base import Blocker
from ..data import KlinkerFrame


class StandardBlocker(Blocker):
    """Fellegi, Ivan P. and Alan B. Sunter. 'A Theory for Record Linkage.' Journal of the American Statistical Association 64 (1969): 1183-1210."""

    def __init__(self, blocking_key: Union[str, List[str]]):
        self.blocking_key = blocking_key

    def _inner_assign(self, kf: KlinkerFrame) -> pd.DataFrame:
        id_col = kf.id_col
        name = kf.name
        blocked = kf[[id_col, self.blocking_key]].groupby(self.blocking_key).agg(list)
        return blocked.rename(columns={id_col: name})

    def _assign(self, left: KlinkerFrame, right: KlinkerFrame) -> pd.DataFrame:
        left_assign = self._inner_assign(left)
        right_assign = self._inner_assign(right)
        return left_assign.join(right_assign, how="outer")
