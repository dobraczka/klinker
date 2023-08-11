from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import dask.bag as db
import pandas as pd
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize

from .base import SchemaAgnosticBlocker
from ..data import KlinkerBlockManager, KlinkerDaskFrame, KlinkerFrame


class MinHashLSHBlocker(SchemaAgnosticBlocker):
    def __init__(
        self,
        tokenize_fn: Callable = word_tokenize,
        threshold: float = 0.5,
        num_perm: int = 128,
        weights: Tuple[float, float] = (0.5, 0.5),
    ):
        self.tokenize_fn = tokenize_fn
        self.threshold = threshold
        self.num_perm = num_perm
        self.weights = weights

    def _inner_encode(self, val: str):
        return [tok.encode("utf-8") for tok in self.tokenize_fn(str(val))]

    def _encode(self, row: Any):
        if isinstance(row, pd.Series):
            res = []
            for _, val in row.items():
                res.extend(self._inner_encode(val))
            return res
        return self._inner_encode(str(row))

    def _create_min_hash_tuple_list(
        self, kf: KlinkerFrame
    ) -> List[Tuple[int, MinHash]]:
        minhash = kf.apply(
            lambda row, id_col, non_id_cols: (
                row[id_col],
                MinHash.bulk([self._encode(row[non_id_cols])])[0],
            ),
            id_col=kf.id_col,
            non_id_cols=kf.non_id_columns,
            axis=1,
        )
        if isinstance(kf, KlinkerDaskFrame):
            minhash = minhash.compute()
        return minhash.tolist()

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> KlinkerBlockManager:
        # for mypy
        assert left.table_name
        assert right.table_name

        block_dict: Dict[Union[str, int], Tuple[Set[int], ...]] = {}
        lsh = MinHashLSH(
            threshold=self.threshold, num_perm=self.num_perm, weights=self.weights
        )
        for number, tab in enumerate([left, right]):
            mh_tuple = self._create_min_hash_tuple_list(tab)

            for row_id, minhash in mh_tuple:
                if number == 0:
                    lsh.insert(row_id, minhash)
                else:
                    res = lsh.query(minhash)
                    if len(res) > 0:
                        block_dict[row_id] = (set(res), {row_id})
        return KlinkerBlockManager.from_dict(block_dict, (left.table_name, right.table_name))
