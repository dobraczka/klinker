from typing import List, Optional

import dask.dataframe as dd
from nltk.util import ngrams

from ..typing import FrameType
from ..data import KlinkerBlockManager
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
        left: FrameType,
        right: FrameType,
        left_rel: Optional[FrameType] = None,
        right_rel: Optional[FrameType] = None,
        left_id_col: str = "head",
        right_id_col: str = "head",
        left_table_name: str = "left",
        right_table_name: str = "right",
    ) -> KlinkerBlockManager:
        """Assign entity ids to blocks.

        Args:
        ----
          left: FrameType: Contains entity attribute information of left dataset.
          right: FrameType: Contains entity attribute information of right dataset.
          left_rel: Optional[FrameType]:  (Default value = None) Contains relational information of left dataset.
          right_rel: Optional[FrameType]:  (Default value = None) Contains relational information of left dataset.

        Returns:
        -------
            KlinkerBlockManager: instance holding the resulting blocks.
        """
        assert isinstance(self.blocking_key, str)
        qgramed = []
        for tab, id_col, table_name in [
            (left, left_id_col, left_table_name),
            (right, right_id_col, right_table_name),
        ]:
            reduced = tab.set_index(id_col)[self.blocking_key]
            if isinstance(left, dd.DataFrame):
                series = reduced.apply(
                    self.qgram_tokenize, meta=(self.blocking_key, "object")
                )
            else:
                series = reduced.apply(self.qgram_tokenize)
            series = series.explode()

            kf = series.to_frame(name=self.blocking_key).reset_index()
            qgramed.append(kf)
        return super().assign(
            left=qgramed[0],
            right=qgramed[1],
            left_id_col=left_id_col,
            right_id_col=right_id_col,
            left_table_name=left_table_name,
            right_table_name=right_table_name,
        )
