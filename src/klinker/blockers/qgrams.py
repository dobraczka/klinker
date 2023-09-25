from typing import Optional

import pandas as pd
from nltk.util import ngrams

from .standard import StandardBlocker
from ..data import KlinkerFrame, KlinkerPandasFrame, KlinkerBlockManager


class QgramsBlocker(StandardBlocker):
    """Blocker relying on qgram procedure

    Args:
        blocking_key: str: On which attribute the blocking should be done
        q: int: how big the qgrams should be.

    Attributes:
        blocking_key: str: On which attribute the blocking should be done
        q: int: how big the qgrams should be.

    """
    def __init__(self, blocking_key: str, q: int = 3):
        super().__init__(blocking_key=blocking_key)
        self.q = q

    def qgram_tokenize(self, x):
        """Tokenize into qgrams

        Args:
          x: input string

        Returns:
            list of qgrams
        """
        if x is None:
            return None
        else:
            return ["".join(tok) for tok in ngrams(x, self.q)]

    def assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Args:
          left: KlinkerFrame: Contains entity attribute information of left dataset.
          right: KlinkerFrame: Contains entity attribute information of right dataset.
          left_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.
          right_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.

        Returns:
            KlinkerBlockManager: instance holding the resulting blocks.
        """
        assert isinstance(self.blocking_key, str)
        qgramed = []
        for tab in [left, right]:
            series = (
                tab.set_index(tab.id_col)[self.blocking_key]
                .apply(self.qgram_tokenize)
                .explode()
            )
            kf = tab.__class__.upgrade_from_series(
                series,
                table_name=tab.table_name,
                id_col=tab.id_col,
                columns=[tab.id_col, self.blocking_key],
            )
            qgramed.append(kf)
        return super().assign(left=qgramed[0], right=qgramed[1])
