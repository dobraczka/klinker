from typing import List, Optional

import dask.dataframe as dd
from nltk.util import ngrams

from ..data import KlinkerBlockManager, KlinkerFrame
from .standard import StandardBlocker


class QgramsBlocker(StandardBlocker):
    """Blocker relying on qgram procedure.

    Args:
    ----
        blocking_key: str: On which attribute the blocking should be done
        q: int: how big the qgrams should be.

    Attributes:
    ----------
        blocking_key: str: On which attribute the blocking should be done
        q: int: how big the qgrams should be.

    Examples:
    --------
        >>> # doctest: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import QgramsBlocker
        >>> blocker = QgramsBlocker(blocking_key="tail")
        >>> blocks = blocker.assign(left=ds.left, right=ds.right)
    """

    def __init__(self, blocking_key: str, q: int = 3):
        super().__init__(blocking_key=blocking_key)
        self.q = q

    def qgram_tokenize(self, x: str) -> Optional[List[str]]:
        """Tokenize into qgrams.

        Args:
        ----
          x: str: input string

        Returns:
        -------
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
        ----
          left: KlinkerFrame: Contains entity attribute information of left dataset.
          right: KlinkerFrame: Contains entity attribute information of right dataset.
          left_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.
          right_rel: Optional[KlinkerFrame]:  (Default value = None) Contains relational information of left dataset.

        Returns:
        -------
            KlinkerBlockManager: instance holding the resulting blocks.
        """
        assert isinstance(self.blocking_key, str)
        qgramed = []
        for tab in [left, right]:
            reduced = tab.set_index(tab.id_col)[self.blocking_key]
            if isinstance(left, dd.DataFrame):
                series = reduced.apply(
                    self.qgram_tokenize, meta=(self.blocking_key, "object")
                )
            else:
                series = reduced.apply(self.qgram_tokenize)
            series = series.explode()

            kf = tab.__class__._upgrade_from_series(
                series,
                table_name=tab.table_name,
                id_col=tab.id_col,
                columns=[tab.id_col, self.blocking_key],
            )
            qgramed.append(kf)
        return super().assign(left=qgramed[0], right=qgramed[1])
