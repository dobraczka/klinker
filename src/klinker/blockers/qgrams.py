from typing import Optional

import pandas as pd
from nltk.util import ngrams

from .standard import StandardBlocker
from ..data import KlinkerPandasFrame, KlinkerFrame


class QgramsBlocker(StandardBlocker):
    def __init__(self, blocking_key: str, q: int = 3):
        super().__init__(blocking_key=blocking_key)
        self.q = q

    def qgram_tokenize(self, x):
        if x is None:
            return None
        else:
            return ["".join(tok) for tok in ngrams(x, self.q)]

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        qgramed = []
        for tab in [left, right]:
            kf = KlinkerPandasFrame(
                tab.set_index(tab.id_col)[self.blocking_key]
                .apply(self.qgram_tokenize)
                .explode(),
                table_name=tab.table_name,
                id_col=tab.id_col,
                columns=[self.blocking_key]
            ).reset_index(inplace=False)
            qgramed.append(kf)
        return super()._assign(left=qgramed[0], right=qgramed[1])
