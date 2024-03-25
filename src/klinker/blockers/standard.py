from typing import Optional

import dask.dataframe as dd
import pandas as pd

from ..data import KlinkerBlockManager, KlinkerDaskFrame, KlinkerFrame
from .base import Blocker


class StandardBlocker(Blocker):
    """Block on same values of a specific column.

    Examples
    --------
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

    def __init__(self, blocking_key: str):
        self.blocking_key = blocking_key

    def _inner_assign(self, kf: KlinkerFrame) -> pd.DataFrame:
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
                    meta=pd.Series(
                        [], dtype=object, index=pd.Index([], name=self.blocking_key)
                    ),
                )
            )
        else:
            series = (
                kf[[id_col, self.blocking_key]]
                .groupby(self.blocking_key)
                .apply(
                    lambda x, id_col: list(set(x[id_col])),
                    id_col=kf.id_col,
                )
            )
        return kf.__class__._upgrade_from_series(
            series,
            columns=[table_name],
            table_name=table_name,
            id_col=id_col,
            reset_index=False,
        )

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Args:
        ----
          left: KlinkerFrame: Contains entity attribute information of left dataset.
          right: KlinkerFrame: Contains entity attribute information of right dataset.
          left_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.
          right_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.

        Returns:
        -------
            KlinkerBlockManager: instance holding the resulting blocks.
        """
        left_assign = self._inner_assign(left)
        right_assign = self._inner_assign(right)
        pd_blocks = left_assign.join(right_assign, how="inner")
        if isinstance(left_assign, dd.DataFrame):
            return KlinkerBlockManager(pd_blocks)
        return KlinkerBlockManager.from_pandas(pd_blocks)
