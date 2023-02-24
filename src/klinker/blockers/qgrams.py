from typing import Iterable

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

    def _assign(self, tables: Iterable[KlinkerFrame]) -> KlinkerFrame:

        qgramed = []
        for tab in tables:
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
                strict_checks=False,
            )
            qgramed.append(kf)
        return super()._assign(qgramed)
