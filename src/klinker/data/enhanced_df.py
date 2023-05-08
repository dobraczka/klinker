import itertools
import numpy as np
from typing import List, Optional, Tuple

import pandas as pd
from pandas._typing import Axes, Dtype

from ..typing import ColumnSpecifier


class KlinkerFrame(pd.DataFrame):
    _metadata = ["name", "id_col"]

    def __init__(
        self,
        data=None,
        index: Optional[Axes] = None,
        columns: Optional[Axes] = None,
        dtype: Optional[Dtype] = None,
        copy: Optional[bool] = None,
        name: Optional[str] = None,
        id_col: Optional[str] = "id",
    ) -> None:
        super().__init__(
            data=data, index=index, columns=columns, dtype=dtype, copy=copy
        )
        self.name = name
        self.id_col = id_col

    @property
    def _constructor(self):
        return KlinkerFrame

    @property
    def prefixed_ids(self) -> pd.Series:
        assert self.name
        return self.name + "_" + self[self.id_col].astype(str)

    @property
    def non_id_columns(self) -> List[str]:
        return [c for c in self.columns if not c == self.id_col]

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, name: str, id_col: Optional[str] = "id"
    ) -> "KlinkerFrame":
        return cls(data=df, name=name, id_col=id_col)

    def concat_values(
        self, columns: ColumnSpecifier = None, new_column_name: str = "_merged_text"
    ) -> "KlinkerFrame":
        if columns is None:
            columns = self.non_id_columns
        wanted = self[columns]
        if isinstance(wanted, pd.Series):
            wanted = wanted.to_frame(name=columns)
        wanted = wanted.astype(str)
        self[new_column_name] = wanted.agg(" ".join, axis=1)
        return self[[self.id_col, new_column_name]]

    def __repr__(self) -> str:
        return super().__repr__() + f"\nTable Name: {self.name}, id_col: {self.id_col}"


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
        assert self.name
        head_with_tail = [self.id_col, self.columns[2]]
        df = (
            self[head_with_tail]
            .groupby(self.id_col)
            .agg(lambda row: " ".join(row.astype(str).values))
            .reset_index()
        )
        df.columns = [self.id_col, new_column_name]
        return KlinkerFrame.from_df(df, name=self.name, id_col=self.id_col)



def combine_blocks(blocks_a: pd.DataFrame, blocks_b: pd.DataFrame) -> pd.DataFrame:
    def _block_merge(
        row: pd.Series,
    ) -> pd.Series:
        def _merge_val(
            row: pd.Series, indices: Tuple[int, int]
        ) -> pd.Series:
            left = row.index[indices[0]]
            right = row.index[indices[1]]
            if row[left] is np.nan:
                return row[right]
            elif row[right] is np.nan:
                return row[left]
            else:
                return row[left] | row[right]

        a = _merge_val(
            row,
            indices=(0,2)
        )
        b = _merge_val(
            row,
            indices=(1,3)
        )
        name_left = row.index[0].replace("left","")
        name_right = row.index[1].replace("left","")
        return pd.Series({name_left: a, name_right: b}, name=row.name)
    return blocks_a.join(blocks_b, how="outer", lsuffix="left",rsuffix="right").apply(_block_merge, axis=1)


@pd.api.extensions.register_dataframe_accessor("klinker_block")
class BlockAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        err_msg = "Only set of ids are allowed as block entries!"
        try:
            if not obj.applymap(lambda x: isinstance(x, set)).all().all():
                raise ValueError(err_msg)
        except AttributeError:
            raise ValueError(err_msg)

    @property
    def block_sizes(self) -> pd.DataFrame:
        return self._obj.apply(
            lambda x: sum((len(v) if isinstance(v, set) else 0 for k, v in x.items())),
            axis=1,
        )

    @property
    def mean_block_size(self) -> float:
        return self.block_sizes.mean()

    def to_pairs(self) -> pd.DataFrame:
        columns = self._obj.columns
        tmp = self._obj.apply(
            lambda row: set(itertools.product(*row.tolist())), axis=1
        ).explode()
        return pd.DataFrame(tmp.tolist(), index=tmp.index, columns=columns)

    def combine(self, other: pd.DataFrame) -> pd.DataFrame:
        return combine_blocks(self._obj, other)
