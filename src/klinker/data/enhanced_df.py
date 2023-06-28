import itertools
from typing import Generator, List, Mapping, Optional, Set, Tuple, Union, Literal, overload

import numpy as np
import pandas as pd
from pandas._typing import Axes, Dtype, IndexLabel

from ..typing import ColumnSpecifier

class KlinkerFrame(pd.DataFrame):
    _metadata = ["table_name", "id_col"]

    def __init__(
        self,
        data=None,
        index: Optional[Axes] = None,
        columns: Optional[Axes] = None,
        dtype: Optional[Dtype] = None,
        copy: Optional[bool] = None,
        table_name: Optional[str] = None,
        id_col: Optional[str] = "id",
    ) -> None:
        super().__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )
        self.table_name = table_name
        self.id_col = id_col

    @property
    def _constructor(self):
        return KlinkerFrame

    @property
    def non_id_columns(self) -> List[str]:
        return [c for c in self.columns if not c == self.id_col]

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, table_name: str, id_col: Optional[str] = "id"
    ) -> "KlinkerFrame":
        return cls(data=df, table_name=table_name, id_col=id_col)

    def concat_values(
        self, columns: ColumnSpecifier = None, new_column_name: str = "_merged_text"
    ) -> "KlinkerFrame":
        if columns is None:
            columns = self.non_id_columns
        wanted = self[columns]
        if isinstance(wanted, pd.Series):
            wanted = wanted.to_frame(name=columns)
        wanted = wanted.astype(str)
        new_df = self.copy()
        new_df[new_column_name] = wanted.agg(" ".join, axis=1)
        return new_df[[self.id_col, new_column_name]]

    # @overload
    # def reset_index(
    #     self,
    #     level: IndexLabel = None,
    #     *,
    #     inplace: Literal[True],
    #     **kwargs,
    # ) -> None:
    #     ...

    # @overload
    # def reset_index(
    #     self,
    #     level: IndexLabel = None,
    #     *,
    #     inplace: Literal[False],
    #     **kwargs,
    # ) -> "KlinkerFrame":
    #     ...

    # def reset_index(
    #     self,
    #     level: IndexLabel = None,
    #     *,
    #     inplace: bool = False,
    #     **kwargs,
    # ) -> Union["KlinkerFrame", None]:
    #     if inplace:
    #         super().reset_index(level, inplace=True, **kwargs)
    #         return None
    #     kf = super().reset_index(level, inplace=False, **kwargs)
    #     assert kf is not None # for mypy
    #     kf.table_name = self.table_name
    #     print(kf)
    #     return kf

    def __repr__(self) -> str:
        return super().__repr__() + f"\nTable Name: {self.table_name}, id_col: {self.id_col}"


class KlinkerTripleFrame(KlinkerFrame):
    @property
    def _constructor(self):
        return KlinkerTripleFrame

    @property
    def non_id_columns(self) -> List[str]:
        return [self.columns[2]]

    def concat_values(
        self, columns: ColumnSpecifier = None, new_column_name: str = "_merged_text"
    ) -> KlinkerFrame:
        assert self.table_name
        head_with_tail = [self.id_col, self.columns[2]]
        df = (
            self.copy()[head_with_tail]
            .groupby(self.id_col)
            .agg(lambda row: " ".join(row.astype(str).values))
            .reset_index()
        )
        df.columns = [self.id_col, new_column_name]
        return KlinkerFrame.from_df(df, table_name=self.table_name, id_col=self.id_col)
