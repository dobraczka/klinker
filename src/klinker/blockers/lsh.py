from typing import Any, Callable, Dict, List, Optional, Tuple

import dask.dataframe as dd
import pandas as pd
from datasketch import LeanMinHash, MinHash, MinHashLSH
from nltk.tokenize import word_tokenize

from .base import SchemaAgnosticBlocker, SeriesType
from ..data import (
    KlinkerBlockManager,
    KlinkerDaskFrame,
    KlinkerFrame,
    generic_upgrade_from_series,
)


def _insert(ser_part: pd.Series, lsh: MinHashLSH, encode_fn: Callable):
    with lsh.insertion_session() as session:
        for key, val in ser_part.items():
            msh = MinHash(num_perm=lsh.h)
            msh.update_batch(encode_fn(val))
            final_msh = LeanMinHash(msh)
            session.insert(key, final_msh)
    return ser_part.index


def _query(
    ser_part: pd.Series,
    lsh: MinHashLSH,
    encode_fn: Callable,
    left_name: str,
    right_name: str,
):
    cur_block: Dict[str, List] = {left_name: [], right_name: []}
    for key, val in ser_part.items():
        msh = MinHash(num_perm=lsh.h)
        msh.update_batch(encode_fn(val))
        final_msh = LeanMinHash(msh)
        res = lsh.query(final_msh)
        if len(res) > 0:
            cur_block[left_name].append(list(res))
            cur_block[right_name].append([key])
    return pd.DataFrame(cur_block)


class MinHashLSHBlocker(SchemaAgnosticBlocker):
    def __init__(
        self,
        tokenize_fn: Callable = word_tokenize,
        threshold: float = 0.5,
        num_perm: int = 128,
        weights: Tuple[float, float] = (0.5, 0.5),
        new_impl: bool = True,
    ):
        self.tokenize_fn = tokenize_fn
        self.threshold = threshold
        self.num_perm = num_perm
        self.weights = weights
        self.new_impl = new_impl

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
        lsh = MinHashLSH(
            threshold=self.threshold,
            num_perm=self.num_perm,
            weights=self.weights,
        )
        if isinstance(left, dd.Series):
            left.map_partitions(
                _insert,
                lsh=lsh,
                encode_fn=self._inner_encode,
                meta=left._meta.index,
            ).compute()
            blocks = right.map_partitions(
                _query,
                lsh=lsh,
                encode_fn=self._inner_encode,
                left_name=left.name,
                right_name=right.name,
                meta=pd.DataFrame([], columns=[left.name, right.name], dtype="O"),
            )
            return KlinkerBlockManager(blocks)
        else:
            _insert(left, lsh=lsh, encode_fn=self._inner_encode)
            blocks = _query(
                right,
                lsh=lsh,
                encode_fn=self._inner_encode,
                left_name=left.name,
                right_name=right.name,
            )
            return KlinkerBlockManager.from_pandas(blocks)
