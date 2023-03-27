from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize

from klinker.data import KlinkerFrame

from .base import Blocker


class MinHashLSHBlocker(Blocker):
    def __init__(
        self,
        blocking_key: Optional[Union[str, List[str]]] = None,
        tokenize_fn: Callable = word_tokenize,
        threshold: float = 0.5,
        num_perm: int = 128,
    ):
        self.blocking_key = blocking_key
        self.tokenize_fn = tokenize_fn
        self.threshold = threshold
        self.num_perm = num_perm

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
        hashed: Dict[str, Dict] = {left.name: {}, right.name: {}}
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        for number, tab in enumerate([left, right]):
            if self.blocking_key is None:
                key = [c for c in tab.columns if not c == tab.id_col]
            else:
                key = self.blocking_key
            tok = tab[key].apply(self._encode, axis=1).tolist()

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
                        hashed[left.name][row_id] = res
                        hashed[right.name][row_id] = [row_id]
        return pd.DataFrame(hashed, columns=[left.name, right.name])
