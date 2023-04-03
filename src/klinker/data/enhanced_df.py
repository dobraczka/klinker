import itertools
from klinker.typing import ColumnSpecifier
from typing import List, Optional

import pandas as pd
from pandas._typing import Axes, Dtype


class KlinkerFrame(pd.DataFrame):
    _metadata = ["name", "id_col"]

    def __init__(
        self,
        data=None,
        index: Optional[Axes] = None,
        columns: Optional[Axes] = None,
        dtype: Optional[Dtype] = None,
        copy: Optional[bool] = None,
        name: str = None,
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

    def concat_values(self, columns: ColumnSpecifier = None, new_column_name: str = "_merged_text") -> "KlinkerFrame":
        if columns is None:
            columns = self.non_id_columns
        wanted = self[columns]
        if isinstance(wanted, pd.Series):
            wanted = wanted.to_frame(name=columns)
        wanted = wanted.astype(str)
        self[new_column_name] = wanted.agg(' '.join, axis=1)
        return self[[self.id_col, new_column_name]]

    def __repr__(self) -> str:
        return super().__repr__() + f"\nTable Name: {self.name}, id_col: {self.id_col}"


class KlinkerTripleFrame(KlinkerFrame):
    @property
    def non_id_columns(self) -> List[str]:
        return [self.columns[2]]

    def concat_values(self, columns: ColumnSpecifier = None, new_column_name: str = "_merged_text") -> KlinkerFrame:
        assert self.name
        new_id_col = "id"
        head_with_tail = [self.id_col, self.columns[2]]
        df = (
            self[head_with_tail]
            .groupby(self.id_col)
            .agg(lambda row: " ".join(row.astype(str).values))
            .reset_index()
        )
        df.columns = [new_id_col, new_column_name]
        return KlinkerFrame.from_df(df, name=self.name, id_col=new_id_col)


@pd.api.extensions.register_dataframe_accessor("klinker_block")
class BlockAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        err_msg = "Only list of ids are allowed as block entries!"
        try:
            if not obj.applymap(lambda x: isinstance(x, list)).all().all():
                raise ValueError(err_msg)
        except:
            raise ValueError(err_msg)

    @property
    def block_sizes(self):
        return self._obj.apply(
            lambda x: sum((len(v) if isinstance(v, list) else 0 for k, v in x.items())),
            axis=1,
        )

    @property
    def mean_block_size(self):
        return self.block_sizes.mean()

    def to_pairs(self):
        columns = self._obj.columns
        tmp = self._obj.apply(
            lambda row: list(itertools.product(*row.tolist())), axis=1
        ).explode()
        return pd.DataFrame(tmp.tolist(), index=tmp.index, columns=columns)
