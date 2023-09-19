from typing import Callable, List, Optional
import logging

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from nltk.tokenize import word_tokenize

from .base import SchemaAgnosticBlocker
from ..data import KlinkerFrame, SeriesType
from ..data.blocks import KlinkerBlockManager
from ..typing import Frame

logger = logging.getLogger(__name__)


def tokenize_series(x, tokenize_fn, min_token_length):
    return set(filter(lambda tok: len(tok) >= min_token_length, tokenize_fn(str(x))))


class TokenBlocker(SchemaAgnosticBlocker):
    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        min_token_length: int = 3,
        intermediate_saving: bool = False,
    ):
        self.tokenize_fn = tokenize_fn
        self.min_token_length = min_token_length
        self.intermediate_saving = intermediate_saving

    def _tok_block(self, tab: SeriesType) -> Frame:
        name = tab.name
        id_col_name = tab.index.name
        # TODO figure out why this hack is needed
        # i.e. why does dask assume later for the join, that this is named 0
        # no matter what it is actually named
        tok_name = 0
        tok_kwargs = dict(
            tokenize_fn=self.tokenize_fn, min_token_length=self.min_token_length
        )
        collect_ids_kwargs = dict(id_col=id_col_name)
        if isinstance(tab, dd.Series):
            tok_kwargs["meta"] = (tab.name, "O")
            collect_ids_kwargs["meta"] = (tab.name, "O")

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

    def _intermediate_saving(
        self, tok_frame: dd.DataFrame, parquet_path: str, index_col_name: str = "tok"
    ):
        name = tok_frame.columns[0]
        tok_frame = tok_frame.reset_index()
        tok_frame._meta = pd.DataFrame([], columns=[0, name], dtype=str)
        tok_frame.rename(columns={0: index_col_name}).set_index(
            index_col_name
        ).to_parquet(parquet_path, schema={name: pa.list_(pa.string())})

    def _assign(
        self,
        left: SeriesType,
        right: SeriesType,
        left_rel: Optional[KlinkerFrame] = None,
        right_rel: Optional[KlinkerFrame] = None,
    ) -> KlinkerBlockManager:
        left_tok = self._tok_block(left)
        right_tok = self._tok_block(right)

        if self.intermediate_saving:
            left_path = "/tmp/left"
            right_path = "/tmp/right"
            self._intermediate_saving(left_tok, left_path)
            logger.info(f"Saved tokenized left as intermediate result in {left_path}")
            self._intermediate_saving(right_tok, right_path)
            logger.info(f"Saved tokenized right as intermediate result in {right_path}")
            left_tok = dd.read_parquet(left_path)
            right_tok = dd.read_parquet(right_path)

        pd_blocks = left_tok.join(right_tok, how="inner")
        if isinstance(pd_blocks, dd.DataFrame):
            return KlinkerBlockManager(pd_blocks)
        return KlinkerBlockManager.from_pandas(pd_blocks)
