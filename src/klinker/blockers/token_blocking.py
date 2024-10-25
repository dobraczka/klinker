import logging
from typing import Dict, Any
from typing import Callable, List, Optional

import dask.dataframe as dd
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from ..data.blocks import KlinkerBlockManager
from ..typing import FrameType
from .base import Blocker
from .concat_utils import is_triple_df, concat_values

logger = logging.getLogger(__name__)


class FilteredTokenizer:
    def __init__(self, tokenize_fn=None, min_token_length=3, stop_words=None):
        if not tokenize_fn:
            tokenize_fn = word_tokenize
        if not stop_words:
            stop_words = stopwords.words("english")
        self.tokenize_fn = tokenize_fn
        self.stop_words = stop_words
        self.min_token_length = min_token_length

    def tokenize(self, x, return_set: bool = False):
        tokens = filter(
            lambda tok: len(tok) >= self.min_token_length
            and tok not in self.stop_words,
            self.tokenize_fn(str(x).lower()),
        )
        if return_set:
            return set(tokens)
        return list(tokens)


class TokenBlocker(Blocker):
    """Tokenizes entity attribute values and blocks on tokens.

    Examples
    --------
        >>> # doctest: +SKIP
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
        stop_words: Optional[List[str]] = None,
        min_token_length: int = 3,
    ):
        self.tokenizer = FilteredTokenizer(
            tokenize_fn=tokenize_fn,
            min_token_length=min_token_length,
            stop_words=stop_words,
        )

    def _create_exploded_token_frame(self, tab, table_name: str):
        tok_kwargs: Dict[str, Any] = {
            "return_set": True,
        }
        if isinstance(tab, dd.Series):
            tok_kwargs["meta"] = (table_name, "O")
        return (
            tab.apply(self.tokenizer.tokenize, **tok_kwargs)
            .explode()
            .dropna()
            .to_frame()
            .reset_index()
        )

    def _tok_block(
        self, tab: FrameType, table_name: str, id_col: str, val_col: str = "tail"
    ) -> FrameType:
        """Perform token blocking on this series.

        Args:
        ----
          tab: SeriesType: series on which token blocking should be done.

        Returns:
        -------
            token blocked series.
        """
        if not isinstance(tab, (pd.Series, dd.Series)):
            if is_triple_df(tab):
                tab = tab.fillna("")
                tab = tab.set_index(id_col)[val_col]
            else:
                tab = concat_values(tab, id_col=id_col)
        tab.name = table_name
        # TODO figure out why this hack is needed
        # i.e. why does dask assume later for the join, that this is named 0
        # no matter what it is actually named
        tok_name = "tok"
        collect_ids_kwargs = {"id_col": id_col}
        if isinstance(tab, dd.Series):
            collect_ids_kwargs["meta"] = pd.Series(
                [],
                name=table_name,
                dtype="O",
                index=pd.Series([], dtype="O", name=tok_name),
            )
        return (
            self._create_exploded_token_frame(tab, table_name)
            .rename(columns={table_name: tok_name})  # avoid same name for col and index
            .groupby(tok_name)
            .apply(lambda x, id_col: list(set(x[id_col])), **collect_ids_kwargs)
            .to_frame(name=table_name)
        )

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
          left: Contains entity attribute information of left dataset.
          right: Contains entity attribute information of right dataset.
          left_rel: Contains relational information of left dataset.
          right_rel: Contains relational information of left dataset.

        Returns:
        -------
            KlinkerBlockManager: instance holding the resulting blocks.
        """
        left_tok = self._tok_block(left, left_table_name, left_id_col)
        right_tok = self._tok_block(right, right_table_name, right_id_col)
        pd_blocks = left_tok.join(right_tok, how="inner")
        if isinstance(pd_blocks, dd.DataFrame):
            return KlinkerBlockManager(pd_blocks)
        return KlinkerBlockManager.from_pandas(pd_blocks)


class UniqueNameBlocker(Blocker):
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
        def filter_nonunique(attr_frame: FrameType, head_col, tail_col):
            if isinstance(attr_frame, pd.DataFrame):
                return attr_frame.groupby(tail_col).filter(
                    lambda x: x[head_col].nunique() == 1
                )
            return attr_frame.groupby(tail_col).apply(
                lambda x: x if x[head_col].nunique() == 1 else None,
                meta=attr_frame._meta,
            )

        if not list(left.columns) == list(right.columns):
            raise ValueError(
                f"Need identical column names but got {left.columns} and {right.columns}"
            )
        head_col = left.columns[0]
        tail_col = left.columns[2]
        left_unique = filter_nonunique(left, head_col, tail_col)
        right_unique = filter_nonunique(right, head_col, tail_col)
        lhead = f"{head_col}_x"
        rhead = f"{head_col}_y"
        res = left_unique.merge(right_unique, on=tail_col, how="inner")[
            [lhead, rhead]
        ].rename(columns={lhead: left.table_name, rhead: right.table_name})
        # avoid problems downstream because of non-str index
        res = res.set_index("unique_" + res.index.astype(str))
        res = res.applymap(lambda x: [x])
        if isinstance(res, pd.DataFrame):
            return KlinkerBlockManager.from_pandas(res)
        return KlinkerBlockManager(res)


if __name__ == "__main__":
    from klinker.data import KlinkerDataset
    from sylloge import MovieGraphBenchmark

    ds = KlinkerDataset.from_sylloge(MovieGraphBenchmark(), clean=True)
    blocks = TokenBlocker().assign(ds.left, ds.right)
