from typing import Optional, Tuple, Union

import pandas as pd

from .base import Blocker
from ..data import KlinkerBlockManager, KlinkerFrame


class StandardBlocker(Blocker):
    """Fellegi, Ivan P. and Alan B. Sunter. 'A Theory for Record Linkage.' Journal of the American Statistical Association 64 (1969): 1183-1210."""

    def __init__(self, blocking_key: Union[str, Tuple[str, str]]):
        self.blocking_key = blocking_key

    def _inner_assign(self, kf: KlinkerFrame, blocking_key: str) -> pd.DataFrame:
        id_col = kf.id_col
        table_name = kf.table_name
        assert table_name

        series = (
            kf[[id_col, self.blocking_key]]
            .groupby(self.blocking_key)
            .apply(lambda x, id_col: set(x[id_col]), id_col=kf.id_col)
        )
        blocked = kf.__class__.upgrade_from_series(
            series,
            columns=[table_name],
            table_name=table_name,
            id_col=id_col,
            reset_index=False,
        )
        return blocked

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> KlinkerBlockManager:
        if isinstance(self.blocking_key, tuple):
            left_bk = self.blocking_key[0]
            right_bk = self.blocking_key[0]
        else:
            left_bk = self.blocking_key
            right_bk = self.blocking_key
        left_assign = self._inner_assign(left, left_bk)
        right_assign = self._inner_assign(right, right_bk)
        pd_blocks = left_assign.join(right_assign, how="inner")
        return KlinkerBlockManager.from_pandas(pd_blocks)
