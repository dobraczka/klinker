from typing import Optional

import dask.dataframe as dd
import pandas as pd

from ..typing import FrameType
from ..data import KlinkerBlockManager
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

    def _inner_assign(self, kf: FrameType, id_col: str, table_name: str) -> FrameType:
        # TODO address code duplication
        if isinstance(kf, dd.DataFrame):
            series = (
                kf[[id_col, self.blocking_key]]
                .groupby(self.blocking_key)
                .apply(
                    lambda x, id_col: list(set(x[id_col])),
                    id_col=id_col,
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
                    id_col=id_col,
                )
            )
        return series.to_frame(name=table_name)

    def assign(
        self,
        left: FrameType,
        right: FrameType,
        left_rel: Optional[FrameType] = None,
        right_rel: Optional[FrameType] = None,
        left_id_col: str = "head",
        right_id_col: str = "head",
        left_table_name: str = "left",
        right_table_name: str = "right",
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Args:
        ----
          left: Contains entity attribute information of left dataset.
          right: Contains entity attribute information of right dataset.
          left_rel: Contains relational information of left dataset.
          right_rel: Contains relational information of left dataset.

        Returns:
        -------
            KlinkerBlockManager: instance holding the resulting blocks.
        """
        left_assign = self._inner_assign(left, left_id_col, left_table_name)
        right_assign = self._inner_assign(right, right_id_col, right_table_name)
        pd_blocks = left_assign.join(right_assign, how="inner")
        if isinstance(left_assign, dd.DataFrame):
            return KlinkerBlockManager(pd_blocks)
        return KlinkerBlockManager.from_pandas(pd_blocks)
