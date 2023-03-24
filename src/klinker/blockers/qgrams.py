from typing import Iterable

import pandas as pd
from nltk.util import ngrams

from klinker.data import KlinkerFrame

from .standard import StandardBlocker


class QgramsBlocker(StandardBlocker):
    def __init__(self, blocking_key: str, q: int = 3):
        super().__init__(blocking_key=blocking_key)
        self.q = q

    def qgram_tokenize(self, x):
        if x is None:
            return None
        else:
            return ["".join(tok) for tok in ngrams(x, self.q)]

    def _assign(self, left: KlinkerFrame, right: KlinkerFrame) -> pd.DataFrame:
        qgramed = []
        for tab in [left,right]:
            data = (
                tab[self.blocking_key]
                .apply(self.qgram_tokenize)
                .explode()
                .to_frame()
                .reset_index()
                .rename(columns={tab.name: self.blocking_key})
            )
            kf = KlinkerFrame(
                data=data,
                name=tab.name,
                id_col="index",
            )
            qgramed.append(kf)
        return super()._assign(left=qgramed[0], right=qgramed[1])
