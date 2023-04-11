from typing import Callable, List, Tuple, Union

import pandas as pd
from nltk.tokenize import word_tokenize

from klinker.blockers import StandardBlocker
from klinker.blockers.base import SchemaAgnosticBlocker
from klinker.data import KlinkerFrame


class TokenBlocker(SchemaAgnosticBlocker):
    def __init__(
        self,
        tokenize_fn: Callable[[str], List[str]] = word_tokenize,
        wanted_cols: Union[
            str, List[str], Tuple[Union[str, List[str]], Union[str, List[str]]]
        ] = None,
    ):
        self.tokenize_fn = tokenize_fn
        super().__init__(wanted_cols=wanted_cols)

    def tokenize(self, x):
        res = []
        for value in x.values:
            res.append(self.tokenize_fn(str(value)))
        return res

    def _assign(self, left: KlinkerFrame, right: KlinkerFrame) -> pd.DataFrame:
        tmp_blocking_key = "_tmp_blocking_key"

        tok_list = []
        for number, tab in enumerate([left, right]):
            tok = (
                tab[tab.non_id_columns]
                .apply(self.tokenize, axis=1)  # returns list of lists
                .explode()  # that's why we need
                .explode()  # 2 explodes
                .to_frame()
                .reset_index()
                .rename(columns={tab.name: tmp_blocking_key})
            )
            tok_list.append(
                KlinkerFrame(
                    data=tok,
                    name=tab.name,
                    id_col="index",
                )
            )
        return StandardBlocker(blocking_key=tmp_blocking_key)._assign(
            tok_list[0], tok_list[1]
        )
