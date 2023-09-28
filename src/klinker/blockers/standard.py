from typing import Optional, Tuple, Union

import dask.dataframe as dd
import pandas as pd

from .base import Blocker
from ..data import KlinkerBlockManager, KlinkerDaskFrame, KlinkerFrame


class StandardBlocker(Blocker):
    """Block on same values of a specific column.

    Examples:

        >>> # doctest: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import StandardBlocker
        >>> blocker = StandardBlocker(blocking_key="tail")
        >>> blocks = blocker.assign(left=ds.left, right=ds.right)

    Quote: Reference
        Fellegi, Ivan P. and Alan B. Sunter. 'A Theory for Record Linkage.' Journal of the American Statistical Association 64 (1969): 1183-1210.
    """

    def __init__(self, blocking_key: Union[str, Tuple[str, str]]):
        self.blocking_key = blocking_key

    def _inner_assign(self, kf: KlinkerFrame, blocking_key: str) -> pd.DataFrame:
        id_col = kf.id_col
        table_name = kf.table_name
        assert table_name

        # TODO address code duplication
        if isinstance(kf, KlinkerDaskFrame):
            series = (
                kf[[id_col, self.blocking_key]]
                .groupby(self.blocking_key)
                .apply(
                    lambda x, id_col: list(set(x[id_col])),
                    id_col=kf.id_col,
                    meta=pd.Series([], dtype=object),
                )
            )
        else:
            series = (
                kf[[id_col, self.blocking_key]]
                .groupby(self.blocking_key)
                .apply(
                    lambda x, id_col: list(set(x[id_col])),
                    id_col=kf.id_col,
                    # TODO add in case dask: meta=pd.Series([], dtype=object),
                )
            )
        blocked = kf.__class__._upgrade_from_series(
            series,
            columns=[table_name],
            table_name=table_name,
            id_col=id_col,
            reset_index=False,
        )
        return blocked

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Args:
          left: KlinkerFrame: Contains entity attribute information of left dataset.
          right: KlinkerFrame: Contains entity attribute information of right dataset.
          left_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.
          right_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.

        Returns:
            KlinkerBlockManager: instance holding the resulting blocks.
        """
        if isinstance(self.blocking_key, tuple):
            left_bk = self.blocking_key[0]
            right_bk = self.blocking_key[0]
        else:
            left_bk = self.blocking_key
            right_bk = self.blocking_key
        left_assign = self._inner_assign(left, left_bk)
        right_assign = self._inner_assign(right, right_bk)
        pd_blocks = left_assign.join(right_assign, how="inner")
        if isinstance(pd_blocks, dd.DataFrame):
            return KlinkerBlockManager(pd_blocks)
        return KlinkerBlockManager.from_pandas(pd_blocks)
