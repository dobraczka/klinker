from abc import ABC, abstractmethod
from typing import List, Optional, Union

import dask.dataframe as dd
import pandas as pd
from dask.dataframe.backends import concat_pandas
from dask.dataframe.core import get_parallel_type
from dask.dataframe.dispatch import make_meta_dispatch
from dask.dataframe.methods import concat_dispatch
from dask.dataframe.utils import meta_nonempty
from pandas._typing import Axes, Dtype

from ..typing import ColumnSpecifier

class AbstractKlinkerFrame(ABC):
    _table_name: Optional[str]
    _id_col: str

    @property
    def table_name(self) -> Optional[str]:
        return self._table_name

    @property
    def id_col(self) -> str:
        return self._id_col

    @property
    @abstractmethod
    def non_id_columns(self) -> List[str]:
        ...

    @abstractmethod
    def concat_values(
        self, columns: ColumnSpecifier = None, new_column_name: str = "_merged_text"
    ):
        ...


class KlinkerPandasFrame(pd.DataFrame, AbstractKlinkerFrame):
    _metadata = ["_table_name", "_id_col"]

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
        assert id_col
        self._table_name = table_name
        self._id_col: str = id_col

    @property
    def _constructor(self):
        return KlinkerPandasFrame

    @property
    def non_id_columns(self) -> List[str]:
        return [c for c in self.columns if not c == self.id_col]

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, table_name: str, id_col: Optional[str] = "id"
    ) -> "KlinkerPandasFrame":
        return cls(data=df, table_name=table_name, id_col=id_col)

    def concat_values(
        self, columns: ColumnSpecifier = None, new_column_name: str = "_merged_text"
    ) -> "KlinkerPandasFrame":
        if columns is None:
            columns = self.non_id_columns
        wanted = self[columns]
        if isinstance(wanted, pd.Series):
            wanted = wanted.to_frame(name=columns)
        wanted = wanted.astype(str)
        new_df = self.copy()
        new_df[new_column_name] = wanted.agg(" ".join, axis=1)
        return new_df[[self.id_col, new_column_name]]

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"\nTable Name: {self.table_name}, id_col: {self.id_col}"
        )


class KlinkerTriplePandasFrame(KlinkerPandasFrame):
    @property
    def _constructor(self):
        return KlinkerTriplePandasFrame

    @property
    def non_id_columns(self) -> List[str]:
        return [self.columns[2]]

    def concat_values(
        self, columns: ColumnSpecifier = None, new_column_name: str = "_merged_text"
    ) -> KlinkerPandasFrame:
        assert self.table_name
        head_with_tail = [self.id_col, self.columns[2]]
        df = (
            self.copy()[head_with_tail]
            .groupby(self.id_col)
            .agg(lambda row: " ".join(row.astype(str).values))
            .reset_index()
        )
        df.columns = [self.id_col, new_column_name]
        return KlinkerPandasFrame.from_df(
            df, table_name=self.table_name, id_col=self.id_col
        )


class KlinkerDaskFrame(dd.core.DataFrame, AbstractKlinkerFrame):
    """Parallel KlinkerFrame

    :param dsk: The dask graph to compute this KlinkerFrame
    :param name: The key prefix that specifies which keys in the dask comprise this particular KlinkerFrame
    :param meta: An empty klinkerframe object with names, dtypes, and indices matching the expected output.
    :param divisions: Values along which we partition our blocks on the index
    """

    _partition_type = KlinkerPandasFrame

    def __init__(
        self,
        dsk,
        name,
        meta,
        divisions,
        table_name: Optional[str] = None,
        id_col: Optional[str] = "id",
    ):
        super().__init__(dsk, name, meta, divisions)
        self._table_name = table_name
        assert id_col
        self._id_col: str = id_col

    @property
    def non_id_columns(self) -> List[str]:
        return [c for c in self.columns if not c == self.id_col]

    def concat_values(
        self, columns: ColumnSpecifier = None, new_column_name: str = "_merged_text"
    ) -> "KlinkerDaskFrame":
        meta = KlinkerPandasFrame(
            pd.DataFrame([], columns=["_merged_text"], dtype="str"),
            table_name="DBpedia",
            id_col="head",
        )
        return self.map_partitions(
            lambda x: x.concat_values,
            columns=columns,
            new_column_name=new_column_name,
            meta=meta,
        )

    @classmethod
    def from_dask_dataframe(
        cls, df: dd.DataFrame, table_name: str, id_col: Optional[str] = "id"
    ) -> "KlinkerDaskFrame":
        new_df = df.map_partitions(
            KlinkerPandasFrame,
            table_name=table_name,
            id_col=id_col,
        )
        return cls(
            dsk=new_df.dask,
            name=new_df._name,
            meta=new_df._meta,
            divisions=new_df.divisions,
            table_name=table_name,
            id_col=id_col,
        )


get_parallel_type.register(KlinkerPandasFrame, lambda _: KlinkerDaskFrame)


@make_meta_dispatch.register(KlinkerPandasFrame)
def make_meta_klinkerpandasframe(df, index=None):
    return df.head(0)


@meta_nonempty.register(KlinkerPandasFrame)
def _nonempty_dataframe(df):
    return KlinkerPandasFrame(
        metanon_empty_dataframe(df), table_name=df.table_name, id_col=df.id_col
    )


@concat_dispatch.register(KlinkerPandasFrame)
def concat_klinker_pandas(
    dfs, axis=0, join="outer", uniform=False, filter_warning=True
):
    return KlinkerPandasFrame(
        concat_pandas(df), table_name=dfs[0].table_name, id_col=dfs[0].id_col
    )

KlinkerFrame = Union[KlinkerPandasFrame, KlinkerDaskFrame]


if __name__ == "__main__":
    from sylloge import OpenEA

    ds = OpenEA(backend="dask")
    tr_left = ds.attr_triples_left
    kdf_left = KlinkerDaskFrame.from_dask_dataframe(
        tr_left, table_name="DBpedia", id_col="head"
    )
    tr_right = ds.attr_triples_right
    kdf_right = KlinkerDaskFrame.from_dask_dataframe(
        tr_right, table_name="Wikidata", id_col="head"
    )
    from klinker.blockers import TokenBlocker

    blocks = TokenBlocker().assign(left=kdf_left, right=kdf_right)

    import ipdb  # noqa: autoimport

    ipdb.set_trace()  # BREAKPOINT
