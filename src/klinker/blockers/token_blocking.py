from typing import Callable, List, Tuple, Union

import pandas as pd
from nltk.tokenize import word_tokenize

from .base import SchemaAgnosticBlocker
from .standard import StandardBlocker
from ..data import KlinkerFrame
from ..utils import tokenize_row


class TokenBlocker(SchemaAgnosticBlocker):
    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        wanted_cols: Union[
            str, List[str], Tuple[Union[str, List[str]], Union[str, List[str]]]
        ] = None,
        min_token_length: int = 3,
    ):
        self.tokenize_fn = tokenize_fn
        super().__init__(wanted_cols=wanted_cols)
        self.min_token_length = min_token_length

    def tokenize(self, x) -> List:
        return tokenize_row(
            x, tokenize_fn=self.tokenize_fn, min_token_length=self.min_token_length
        )

    def _assign(self, left: KlinkerFrame, right: KlinkerFrame) -> pd.DataFrame:
        tmp_blocking_key = "_tmp_blocking_key"

        tok_list = []
        for tab in [left, right]:
            tok = (
                tab.set_index(tab.id_col)[tab.non_id_columns]
                .apply(self.tokenize, axis=1)
                .explode()
                .to_frame()
                .reset_index()
                .rename(columns={tab.name: tmp_blocking_key})
            )
            tok_list.append(
                KlinkerFrame(
                    data=tok,
                    name=tab.name,
                    id_col=tab.id_col,
                )
            )
        return StandardBlocker(blocking_key=tmp_blocking_key)._assign(
            tok_list[0], tok_list[1]
        )
