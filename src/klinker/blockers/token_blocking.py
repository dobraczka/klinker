from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
from nltk.tokenize import word_tokenize

from .base import SchemaAgnosticBlocker
from .standard import StandardBlocker
from ..data import KlinkerFrame, KlinkerPandasFrame
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

    def _assign(
        self,
        left: KlinkerFrame,
        right: KlinkerFrame,
        left_rel: Optional[pd.DataFrame] = None,
        right_rel: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        tmp_blocking_key = "_tmp_blocking_key"

        tok_list: List[KlinkerFrame] = []
        for tab in [left, right]:
            print(tab)
            kf = KlinkerPandasFrame(
                tab.set_index(tab.id_col)[tab.non_id_columns]
                .apply(self.tokenize, axis=1)
                .explode(),
                table_name=tab.table_name,
                id_col=tab.id_col,
                columns=[tmp_blocking_key],
            ).reset_index(inplace=False)
            tok_list.append(kf)
        print(tok_list)
        return StandardBlocker(blocking_key=tmp_blocking_key)._assign(
            tok_list[0], tok_list[1]
        )
