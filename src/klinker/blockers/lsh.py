from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import pandas as pd
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize

from .base import SchemaAgnosticBlocker, SeriesType
from ..data import (
    KlinkerBlockManager,
    KlinkerDaskFrame,
    KlinkerFrame,
    generic_upgrade_from_series,
)


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
        self, conc: SeriesType
    ) -> List[Tuple[str, MinHash]]:
        frame_class: Type[KlinkerFrame]
        kf = generic_upgrade_from_series(conc, reset_index=True)

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
        left: SeriesType,
        right: SeriesType,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        block_dict: Dict[str, Tuple[List[str], List[str]]] = {}
        lsh = MinHashLSH(
            threshold=self.threshold, num_perm=self.num_perm, weights=self.weights
        )
        for number, tab in enumerate([left, right]):
            mh_tuple = self._create_min_hash_tuple_list(tab)

            for row_id, minhash in mh_tuple:
                if number == 0:
                    lsh.insert(row_id, minhash)
                else:
                    res = list(set(lsh.query(minhash)))
                    if len(res) > 0:
                        block_dict[row_id] = (res, [row_id])

        return KlinkerBlockManager.from_dict(block_dict, (left.name, right.name))
