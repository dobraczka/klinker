from collections import defaultdict
from typing import Dict, List, Optional, Set, Union, Tuple

import pandas as pd

from .base import Blocker
from ..data import KlinkerBlockManager, KlinkerFrame


class StandardBlocker(Blocker):
    """Fellegi, Ivan P. and Alan B. Sunter. 'A Theory for Record Linkage.' Journal of the American Statistical Association 64 (1969): 1183-1210."""

    def __init__(self, blocking_key: Union[str, List[str]]):
        self.blocking_key = blocking_key

    def _inner_assign(self, kf: KlinkerFrame) -> pd.DataFrame:
        id_col = kf.id_col
        name = kf.name
        blocked = kf[[id_col, self.blocking_key]].groupby(self.blocking_key).agg(set)
        return blocked.rename(columns={id_col: name})

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> KlinkerBlockManager:
        left_assign = self._inner_assign(left)
        right_assign = self._inner_assign(right)
        pd_blocks = left_assign.join(right_assign, how="inner")
        return KlinkerBlockManager.from_pandas(pd_blocks)


class NewStandardBlocker(Blocker):
    def __init__(self, blocking_key: Union[str, List[str]]):
        self.blocking_key = blocking_key

    def _assign_side(
        self,
        kf: KlinkerFrame,
        block_dict: Dict[Union[str, int], Tuple[Set[int], ...]],
        side_int: int,
    ) -> Dict[Union[str, int], Tuple[Set[int], ...]]:
        for _, id_blk in kf[[kf.id_col, self.blocking_key]].iterrows():
            block_dict[id_blk[self.blocking_key]][side_int].add(id_blk[kf.id_col])
        return block_dict

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # for mypy
        assert left.name
        assert right.name
        block_dict: Dict[Union[str, int], Tuple[Set[int], ...]] = defaultdict(
            lambda: (set(), set())
        )
        block_dict = self._assign_side(left, block_dict, 0)
        block_dict = self._assign_side(right, block_dict, 1)
        return KlinkerBlockManager(block_dict, (left.name, right.name))


if __name__ == "__main__":
    from sylloge import MovieGraphBenchmark

    from klinker.data import KlinkerDataset
    from time import time

    data = KlinkerDataset.from_sylloge(MovieGraphBenchmark(), clean=True)
    start = time()
    block = StandardBlocker(blocking_key="tail").assign(data.left, data.right)
    end = time() - start
    print(end)
    start = time()
    block = NewStandardBlocker(blocking_key="tail").assign(data.left, data.right)
    end = time() - start
    print(end)
