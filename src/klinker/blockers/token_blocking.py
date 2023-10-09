import logging
from typing import Callable, List, Optional

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from nltk.tokenize import word_tokenize

from .base import SchemaAgnosticBlocker
from ..data import KlinkerFrame
from ..data.blocks import KlinkerBlockManager
from ..typing import Frame, SeriesType

logger = logging.getLogger(__name__)


def tokenize_series(x, tokenize_fn, min_token_length):
    """Tokenize a series and return set.

    Args:
      x: series with values to tokenize
      tokenize_fn: tokenization function
      min_token_length: minimum length of tokens

    Returns:
        set of tokens
    """
    return set(filter(lambda tok: len(tok) >= min_token_length, tokenize_fn(str(x))))


class TokenBlocker(SchemaAgnosticBlocker):
    """Concatenates and tokenizes entity attribute values and blocks on tokens.

    Examples:

        >>> # doctests: +SKIP
        >>> from sylloge import MovieGraphBenchmark
        >>> from klinker.data import KlinkerDataset
        >>> ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(),clean=True)
        >>> from klinker.blockers import TokenBlocker
        >>> blocker = TokenBlocker()
        >>> blocks = blocker.assign(left=ds.left, right=ds.right)

    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        min_token_length: int = 3,
    ):
        self.tokenize_fn = tokenize_fn
        self.min_token_length = min_token_length

    def _tok_block(self, tab: SeriesType) -> Frame:
        """Perform token blocking on this series.

        Args:
          tab: SeriesType: series on which token blocking should be done.

        Returns:
            token blocked series.
        """
        name = tab.name
        id_col_name = tab.index.name
        # TODO figure out why this hack is needed
        # i.e. why does dask assume later for the join, that this is named 0
        # no matter what it is actually named
        tok_name = "tok"
        tok_kwargs = dict(
            tokenize_fn=self.tokenize_fn, min_token_length=self.min_token_length
        )
        collect_ids_kwargs = dict(id_col=id_col_name)
        if isinstance(tab, dd.Series):
            tok_kwargs["meta"] = (tab.name, "O")
            collect_ids_kwargs["meta"] = pd.Series(
                [],
                name=tab.name,
                dtype="O",
                index=pd.Series([], dtype="O", name=tok_name),
            )
        return (
            tab.apply(tokenize_series, **tok_kwargs)
            .explode()
            .to_frame()
            .reset_index()
            .rename(columns={name: tok_name})  # avoid same name for col and index
            .groupby(tok_name)
            .apply(lambda x, id_col: list(set(x[id_col])), **collect_ids_kwargs)
            .to_frame(name=name)
        )

    def _assign(
        self,
        left: SeriesType,
        right: SeriesType,
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
        left_tok = self._tok_block(left)
        right_tok = self._tok_block(right)
        pd_blocks = left_tok.join(right_tok, how="inner")
        if isinstance(pd_blocks, dd.DataFrame):
            return KlinkerBlockManager(pd_blocks)
        return KlinkerBlockManager.from_pandas(pd_blocks)
