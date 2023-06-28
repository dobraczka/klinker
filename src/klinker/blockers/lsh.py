from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
from collections import defaultdict
import dask.bag as db

import pandas as pd
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize

from .base import SchemaAgnosticBlocker
from ..data import KlinkerFrame, KlinkerBlockManager


class MinHashLSHBlocker(SchemaAgnosticBlocker):
    def __init__(
        self,
        tokenize_fn: Callable = word_tokenize,
        threshold: float = 0.5,
        num_perm: int = 128,
        weights: Tuple[float, float] = (0.5, 0.5),
        wanted_cols: Union[
            str, List[str], Tuple[Union[str, List[str]], Union[str, List[str]]]
        ] = None,
    ):
        self.tokenize_fn = tokenize_fn
        self.threshold = threshold
        self.num_perm = num_perm
        self.weights = weights
        super().__init__(wanted_cols=wanted_cols)

    def _inner_encode(self, val: str):
        return [tok.encode("utf-8") for tok in self.tokenize_fn(str(val))]

    def _encode(self, row: Any):
        if isinstance(row, pd.Series):
            res = []
            for _, val in row.items():
                res.extend(self._inner_encode(val))
            return res
        return self._inner_encode(str(row))

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
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm, weights=self.weights)
        for number, tab in enumerate([left, right]):
            tok = tab[tab.non_id_columns].apply(self._encode, axis=1).tolist()

            for minhash, row_id in zip(
                MinHash.generator(tok, num_perm=self.num_perm),
                tab[tab.id_col],
            ):
                if number == 0:
                    lsh.insert(row_id, minhash)
                else:
                    res = lsh.query(minhash)
                    if len(res) > 0:
                        block_dict[row_id] = (set(res), {row_id})
        return KlinkerBlockManager(block_dict, (left.table_name, right.table_name))


class NewMinHashLSHBlocker(SchemaAgnosticBlocker):
    def __init__(
        self,
        tokenize_fn: Callable = word_tokenize,
        threshold: float = 0.5,
        num_perm: int = 128,
        weights: Tuple[float, float] = (0.5, 0.5),
        wanted_cols: Union[
            str, List[str], Tuple[Union[str, List[str]], Union[str, List[str]]]
        ] = None,
    ):
        self.tokenize_fn = tokenize_fn
        self.threshold = threshold
        self.num_perm = num_perm
        self.weights = weights
        super().__init__(wanted_cols=wanted_cols)

    def _inner_encode(self, val: str):
        return [tok.encode("utf-8") for tok in self.tokenize_fn(str(val))]

    def _encode(self, row: Any):
        if isinstance(row, pd.Series):
            res = []
            for _, val in row.items():
                res.extend(self._inner_encode(val))
            return res
        return self._inner_encode(str(row))

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        left_name = left.table_name
        right_name = right.table_name
        # for mypy
        assert left_name
        assert right_name
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm, weights=self.weights)
        block_dict: Dict[Union[str, int], Tuple[Set[int], ...]] = {}

        mybagleft = db.from_sequence(left["_merged_text"].items())
        mybagright = db.from_sequence(right["_merged_text"].items())

        for myhash in mybagleft.map(lambda row: [(row[0], mhash) for mhash in MinHash.generator([(tok.encode("utf-8") for tok in self.tokenize_fn(row[1]))])]).flatten().compute():
            lsh.insert(myhash[0], myhash[1])

        for myhash in mybagright.map(lambda row: [(row[0], mhash) for mhash in MinHash.generator([(tok.encode("utf-8") for tok in self.tokenize_fn(row[1]))])]).flatten().compute():
            res = lsh.query(myhash[1])
            if len(res) > 0:
                block_dict[myhash[0]] = (set(res), {myhash[0]})
        return KlinkerBlockManager(block_dict, (left_name, right_name))
