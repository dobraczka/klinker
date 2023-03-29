from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize

from klinker.data import KlinkerFrame

from .base import SchemaAgnosticBlocker


class MinHashLSHBlocker(SchemaAgnosticBlocker):
    def __init__(
        self,
        tokenize_fn: Callable = word_tokenize,
        threshold: float = 0.5,
        num_perm: int = 128,
        wanted_cols: Union[
            str, List[str], Tuple[Union[str, List[str]], Union[str, List[str]]]
        ] = None,
    ):
        self.tokenize_fn = tokenize_fn
        self.threshold = threshold
        self.num_perm = num_perm
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

    def _assign(self, left: KlinkerFrame, right: KlinkerFrame) -> pd.DataFrame:
        left_name = left.name
        right_name = right.name
        # for mypy
        assert left_name
        assert right_name

        hashed: Dict[str, Dict] = {left_name: {}, right_name: {}}
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        for number, tab in enumerate([left, right]):
            tok = (
                tab[self._actual_wanted_cols[number]]
                .apply(self._encode, axis=1)
                .tolist()
            )

            for minhash, row_id in zip(
                MinHash.generator(tok),
                tab[tab.id_col],
            ):
                f"{tab.name}_{row_id}"
                if number == 0:
                    lsh.insert(row_id, minhash)
                else:
                    res = lsh.query(minhash)
                    if len(res) > 0:
                        hashed[left_name][row_id] = res
                        hashed[right_name][row_id] = [row_id]
        return pd.DataFrame(hashed, columns=[left.name, right.name])
