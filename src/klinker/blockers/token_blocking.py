from typing import Callable, List, Optional

import dask.bag as db
import pandas as pd
from nltk.tokenize import word_tokenize

from .base import SchemaAgnosticBlocker
from .standard import StandardBlocker
from ..data import KlinkerDaskFrame, KlinkerFrame
from ..data.blocks import KlinkerBlockManager
from ..utils import tokenize_row


class TokenBlocker(SchemaAgnosticBlocker):
    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        min_token_length: int = 3,
        use_alternative_dask_impl: bool = False,
    ):
        self.tokenize_fn = tokenize_fn
        self.min_token_length = min_token_length
        self.use_alternative_dask_impl = use_alternative_dask_impl

    def tokenize(self, x) -> List:
        return tokenize_row(
            x, tokenize_fn=self.tokenize_fn, min_token_length=self.min_token_length
        )

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> KlinkerBlockManager:
        tmp_blocking_key = "_tmp_blocking_key"

        tok_list: List[KlinkerFrame] = []
        for tab in [left, right]:
            kf = tab.__class__.upgrade_from_series(
                tab.set_index(tab.id_col)[tab.non_id_columns]
                .apply(self.tokenize, axis=1)
                .explode(),
                table_name=tab.table_name,
                id_col=tab.id_col,
                columns=[tab.id_col, tmp_blocking_key],
                reset_index=True,
            )
            tok_list.append(kf)
        return StandardBlocker(blocking_key=tmp_blocking_key).assign(
            tok_list[0], tok_list[1]
        )
