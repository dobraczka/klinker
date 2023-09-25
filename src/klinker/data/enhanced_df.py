from abc import ABC, abstractmethod
from typing import List, Optional, Type, Union

import dask.dataframe as dd
import pandas as pd
from dask.core import no_default
from dask.dataframe.backends import concat_pandas, meta_nonempty_dataframe
from dask.dataframe.core import get_parallel_type
from dask.dataframe.dispatch import make_meta_dispatch
from dask.dataframe.methods import concat_dispatch
from dask.dataframe.utils import meta_nonempty
from dask.utils import M
from pandas._typing import Axes, Dtype

SeriesType = Union[pd.Series, dd.Series]

class AbstractKlinkerFrame(ABC):
    """Abstract klinker frame class"""
    _table_name: Optional[str]
    _id_col: str

    @property
    def table_name(self) -> Optional[str]:
        """Name of dataset"""
        return self._table_name

    @table_name.setter
    def table_name(self, value: str):
        self._table_name = value

    @property
    def id_col(self) -> str:
        """Column where entity ids are stored"""
        return self._id_col

    @id_col.setter
    def id_col(self, value: str):
        self._id_col = value

    @property
    @abstractmethod
    def non_id_columns(self) -> List[str]:
        """Other columns than column with entity ids"""
        ...

    @abstractmethod
    def concat_values(
        self,
    ) -> SeriesType:
        """Concatenated entity attribute values.

        Returns:
            Concatenated attribute values as series with ids as index.
        """
        ...

    @classmethod
    @abstractmethod
    def _upgrade_from_series(
        cls,
        series: SeriesType,
        columns: List[str],
        table_name: Optional[str],
        id_col: str,
        reset_index: bool = True,
    ) -> "KlinkerFrame":
        """Upgrade series to KlinkerFrame.

        Args:
          series: SeriesType: series to upgrade
          columns: List[str]: column names of resulting df
          table_name: Optional[str]: dataset name
          id_col: str: name of id column
          reset_index: bool: whether to make id_col a seperate column

        Returns:
            klinker dataframe
        """
        ...


class KlinkerPandasFrame(pd.DataFrame, AbstractKlinkerFrame):
    """Enhanced pandas Dataframe for klinker.

    This keeps `table_name` and `id_col` as metadata
    throughout transformations as best as possible.

    Furthermore specific methods for blocking are implemented.

    Examples:
        >>> import pandas as pd
        >>> from klinker.data import KlinkerPandasFrame
        >>> df = pd.DataFrame([("1","John", "Doe"),("2","Jane","Doe")],columns=["id","first name", "surname"])
        >>> df
          id first name surname
        0  1       John     Doe
        1  2       Jane     Doe
        >>> kdf = KlinkerPandasFrame.from_df(df, table_name="A", id_col="id")
        >>> kdf
          id first name surname
        0  1       John     Doe
        1  2       Jane     Doe
        Table Name: A, id_col: id
        >>> kdf.non_id_columns
        ['first name', 'surname']
        >>> kdf.concat_values()
        id
        1    John Doe
        2    Jane Doe
        Name: A, dtype: object
    """
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
        """ """
        return KlinkerPandasFrame

    @property
    def non_id_columns(self) -> List[str]:
        """ """
        return [c for c in self.columns if not c == self.id_col]

    @classmethod
    def from_df(
        cls, df: pd.DataFrame, table_name: str, id_col: Optional[str] = "id"
    ) -> "KlinkerPandasFrame":
        """Construct a KlinkerPandasFrame from a pd.DataFrame.

        Args:
          df: pd.DataFrame: The df holding the data
          table_name: str: Name of the dataset.
          id_col: Optional[str]:  Column with entity ids ("id" as default).

        Returns:
            KlinkerPandasFrame

        Examples:
            >>> import pandas as pd
            >>> from klinker.data import KlinkerPandasFrame
            >>> df = pd.DataFrame([("1","John", "Doe"),("2","Jane","Doe")],columns=["id","first name", "surname"])
            >>> kdf = KlinkerPandasFrame.from_df(df, table_name="A", id_col="id")
            >>> kdf
              id first name surname
            0  1       John     Doe
            1  2       Jane     Doe
            Table Name: A, id_col: id
        """
        return cls(data=df, table_name=table_name, id_col=id_col)

    def concat_values(
        self,
    ) -> pd.Series:
        """Concatenate all values, that are not in the id_col.

        Returns:
            Series with id_col as index and concatenated values.

        Examples:
            >>> import pandas as pd
            >>> from klinker.data import KlinkerPandasFrame
            >>> df = pd.DataFrame([("1","John", "Doe"),("2","Jane","Doe")],columns=["id","first name", "surname"])
            >>> kdf = KlinkerPandasFrame.from_df(df, table_name="A", id_col="id")
            >>> kdf.concat_values()
            id
            1    John Doe
            2    Jane Doe
            Name: A, dtype: object
        """
        self = self.fillna("")
        result = (
            self.copy()
            .set_index(self.id_col)[self.non_id_columns]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.strip()
        )
        result.name = self.table_name
        return result

    @classmethod
    def _upgrade_from_series(
        cls,
        series,
        columns: List[str],
        table_name: Optional[str],
        id_col: str,
        reset_index: bool = True,
    ) -> "KlinkerFrame":
        kf = KlinkerPandasFrame(series.to_frame(), table_name=table_name, id_col=id_col)
        if reset_index:
            kf = kf.reset_index()
        kf.columns = columns
        return kf

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"\nTable Name: {self.table_name}, id_col: {self.id_col}"
        )


class KlinkerTriplePandasFrame(KlinkerPandasFrame):
    """Class for holding triple information.

    Example:
        >>> import pandas as pd
        >>> from klinker.data import KlinkerTriplePandasFrame
        >>> df = pd.DataFrame([("e1","foaf:givenname","John"),("e1","foaf:family_name", "Doe"), ("e2","rdfs:label","Jane Doe")],columns=["head","rel","tail"])
        >>> from klinker.data import KlinkerTriplePandasFrame
        >>> kdf = KlinkerTriplePandasFrame.from_df(df, table_name="A",id_col="head")
        >>> kdf
        head               rel      tail
        0   e1    foaf:givenname      John
        1   e1  foaf:family_name       Doe
        2   e2        rdfs:label  Jane Doe
        Table Name: A, id_col: head
        >>> kdf.concat_values()
        head
        e1    John Doe
        e2    Jane Doe
        Name: A, dtype: object
    """
    @property
    def _constructor(self):
        """ """
        return KlinkerTriplePandasFrame

    @property
    def non_id_columns(self) -> List[str]:
        """Last column."""
        return [self.columns[2]]

    def concat_values(
        self,
    ) -> pd.Series:
        """Concatenate all values of the tail column.

        Returns:
            Series with id_col as index and concatenated values.

        Example:
            >>> import pandas as pd
            >>> from klinker.data import KlinkerTriplePandasFrame
            >>> df = pd.DataFrame([("e1","foaf:givenname","John"),("e1","foaf:family_name", "Doe"), ("e2","rdfs:label","Jane Doe")],columns=["head","rel","tail"])
            >>> from klinker.data import KlinkerTriplePandasFrame
            >>> kdf = KlinkerTriplePandasFrame.from_df(df, table_name="A",id_col="head")
            >>> kdf.concat_values()
            head
            e1    John Doe
            e2    Jane Doe
            Name: A, dtype: object
        """
        assert self.table_name
        self = self.fillna("")
        res = (
            self[[self.id_col, self.columns[2]]]
            .groupby(self.id_col)
            .agg(lambda row: " ".join(row.astype(str).values).strip())[self.columns[2]]
        )
        res.name = self.table_name
        return res


class KlinkerDaskFrame(dd.core.DataFrame, AbstractKlinkerFrame):
    """Parallel KlinkerFrame.

    Please don't use the `__init__` method but rather `from_dask_dataframe` for
    initialisation!

    Args:
      dsk: The dask graph to compute this KlinkerFrame
      name: The key prefix that specifies which keys in the dask comprise this particular KlinkerFrame
      meta: An empty klinkerframe object with names, dtypes, and indices matching the expected output.
      divisions: Values along which we partition our blocks on the index

    Returns:
        KlinkerDaskFrame

    Example:
        >>> import pandas as pd
        >>> from klinker.data import KlinkerDaskFrame
        >>> import dask.dataframe as dd
        >>> df = dd.from_pandas(pd.DataFrame([("1","John", "Doe"),("2","Jane","Doe")],columns=["id","first name", "surname"]),npartitions=1)
        >>> kdf = KlinkerDaskFrame.from_dask_dataframe(df, table_name="A", id_col="id")
        >>> kdf
        Dask KlinkerDaskFrame Structure:
                           id first name surname
        npartitions=1
        0              object     object  object
        1                 ...        ...     ...
        Dask Name: KlinkerPandasFrame, 2 graph layers
        Table Name: A, id_col: id
    """

    _partition_type = KlinkerPandasFrame

    def __init__(
        self,
        dsk,
        name,
        meta,
        divisions,
        table_name: Optional[str] = None,
        id_col: str = "id",
    ):
        super().__init__(dsk, name, meta, divisions)
        if table_name is None:
            self._table_name = meta.table_name
            self._id_col = meta.id_col
        else:
            self._table_name = table_name
            self._id_col = id_col

    @staticmethod
    def _static_propagate_klinker_attributes(
        new_object: "KlinkerDaskFrame", table_name: str, id_col: str
    ) -> "KlinkerDaskFrame":
        new_object.table_name = table_name
        new_object.id_col = id_col
        return new_object

    @property
    def non_id_columns(self) -> List[str]:
        """All columns which are not `id_col`"""
        return self._meta.non_id_columns

    @classmethod
    def _upgrade_from_series(
        cls,
        series,
        columns: List[str],
        table_name: Optional[str],
        id_col: str,
        reset_index: bool = True,
        meta=no_default,
    ) -> "KlinkerFrame":
        assert table_name
        kf = series.map_partitions(
            KlinkerPandasFrame.upgrade_from_series,
            columns=columns,
            table_name=table_name,
            id_col=id_col,
            reset_index=reset_index,
            meta=meta,
        )
        return KlinkerDaskFrame._static_propagate_klinker_attributes(
            kf, table_name, id_col
        )

    def concat_values(
        self,
    ) -> dd.Series:
        """Concatenate attribute values.

        Returns:
            dd.Series with concatenated values and id_col as index.

        Example:
            >>> import pandas as pd
            >>> from klinker.data import KlinkerDaskFrame
            >>> import dask.dataframe as dd
            >>> df = dd.from_pandas(pd.DataFrame([("1","John", "Doe"),("2","Jane","Doe")],columns=["id","first name", "surname"]),npartitions=1)
            >>> kdf = KlinkerDaskFrame.from_dask_dataframe(df, table_name="A", id_col="id")
            >>> kdf.concat_values().compute()
            id
            1    John Doe
            2    Jane Doe
            Name: A, dtype: object
        """
        self = self.fillna("")
        assert self.table_name
        meta = pd.Series([], name=self.table_name, dtype="str")
        meta.index.name = self.id_col
        return self.map_partitions(
            M.concat_values,
            meta=meta,
        )

    @classmethod
    def from_dask_dataframe(
        cls,
        df: dd.DataFrame,
        table_name: str,
        id_col: str,
        meta=no_default,
        construction_class: Type[KlinkerPandasFrame] = KlinkerPandasFrame,
    ) -> "KlinkerDaskFrame":
        """Create KlinkDaskFrame from dask dataframe.

        Args:
          df: dd.DataFrame: Dask dataframe.
          table_name: str: Name of dataset.
          id_col: str: Column where entity_ids are stored
          meta: meta for dask
          construction_class: Either :class:`KlinkerPandasFrame` or :class:`KlinkerTriplePandasFrame`

        Returns:
            KlinkerDaskFrame

        Example:
            >>> import pandas as pd
            >>> from klinker.data import KlinkerDaskFrame
            >>> import dask.dataframe as dd
            >>> df = dd.from_pandas(pd.DataFrame([("1","John", "Doe"),("2","Jane","Doe")],columns=["id","first name", "surname"]),npartitions=1)
            >>> kdf = KlinkerDaskFrame.from_dask_dataframe(df, table_name="A", id_col="id")
            >>> kdf
            Dask KlinkerDaskFrame Structure:
                               id first name surname
            npartitions=1
            0              object     object  object
            1                 ...        ...     ...
            Dask Name: KlinkerPandasFrame, 2 graph layers
            Table Name: A, id_col: id
        """
        new_df = df.map_partitions(
            construction_class,
            table_name=table_name,
            id_col=id_col,
            meta=meta,
        )
        meta = new_df._meta if meta is no_default else meta
        return cls(
            dsk=new_df.dask,
            name=new_df._name,
            meta=meta,
            divisions=new_df.divisions,
            table_name=table_name,
            id_col=id_col,
        )

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"\nTable Name: {self.table_name}, id_col: {self.id_col}"
        )


class KlinkerTripleDaskFrame(KlinkerDaskFrame):
    """Parallel KlinkerTriplePandasFrame

    Args:
      dsk: The dask graph to compute this KlinkerFrame
      name: The key prefix that specifies which keys in the dask comprise this particular KlinkerFrame
      meta: An empty klinkerframe object with names, dtypes, and indices matching the expected output.
      divisions: Values along which we partition our blocks on the index

    Returns:

    """

    _partition_type = KlinkerTriplePandasFrame

    def concat_values(
        self,
    ) -> dd.Series:
        """


        Returns:

        """
        self = self.fillna("")
        assert self.table_name
        result = self.groupby(self.id_col)[self.columns[2]].apply(
            lambda grp: " ".join(grp.astype(str)).strip(),
            meta=pd.Series([], name=self.columns[2], dtype="str"),
        )
        result.name = self.table_name
        result._meta.index.name = self.id_col
        return result


def from_klinker_frame(kf: KlinkerPandasFrame, npartitions: int) -> "KlinkerDaskFrame":
    """Create KlinkerDaskFrame from KlinkerPandasFrame.

    Args:
      kf: KlinkerPandasFrame: Input dataframe
      npartitions: int: Number of partitions for dask.

    Returns:
        KlinkerDaskFrame
    """
    if not kf.table_name:
        raise ValueError("KlinkerFrame needs to have a table_name set!")
    cls = (
        KlinkerTripleDaskFrame
        if isinstance(kf, KlinkerTriplePandasFrame)
        else KlinkerDaskFrame
    )
    return cls.from_dask_dataframe(
        dd.from_pandas(kf, npartitions=npartitions),
        table_name=kf.table_name,
        id_col=kf.id_col,
        meta=kf.head(0),
        construction_class=kf.__class__,
    )

#====== dask related ======
get_parallel_type.register(KlinkerPandasFrame, lambda _: KlinkerDaskFrame)
get_parallel_type.register(KlinkerTriplePandasFrame, lambda _: KlinkerTripleDaskFrame)


@make_meta_dispatch.register((KlinkerPandasFrame, KlinkerTriplePandasFrame))
def make_meta_klinkerpandasframe(df, index=None):
    return df.head(0)


@meta_nonempty.register((KlinkerPandasFrame, KlinkerTriplePandasFrame))
def _nonempty_dataframe(df):
    return KlinkerPandasFrame(
        meta_nonempty_dataframe(df), table_name=df.table_name, id_col=df.id_col
    )


@concat_dispatch.register(KlinkerPandasFrame)
def concat_klinker_pandas(
    dfs,
    axis=0,
    join="outer",
    uniform=False,
    filter_warning=True,
    ignore_index=False,
    **kwargs,
):
    return KlinkerPandasFrame(
        concat_pandas(dfs), table_name=dfs[0].table_name, id_col=dfs[0].id_col
    )


@concat_dispatch.register(KlinkerTriplePandasFrame)
def concat_klinker_triple_pandas(
    dfs,
    axis=0,
    join="outer",
    uniform=False,
    filter_warning=True,
    ignore_index=False,
    **kwargs,
):
    return KlinkerTriplePandasFrame(
        concat_pandas(dfs), table_name=dfs[0].table_name, id_col=dfs[0].id_col
    )

#====== end dask related ======

KlinkerFrame = Union[KlinkerPandasFrame, KlinkerDaskFrame]


def generic_upgrade_from_series(conc: SeriesType, reset_index: bool = False) -> KlinkerFrame:
    """Upgrade a series to KlinkerFrame.

    This automatically determines the correct KlinkerFrame class
    based on the given series class.

    Note:
        This will use the series name as the resulting dataset name.
        The series index is assumed to be the entity ids.

    Args:
      conc: SeriesType: Series to upgrade.
      reset_index: bool: If True resets index.

    Returns:
        KlinkerFrame

    Example:
        >>> import pandas as pd
        >>> from klinker.data import generic_upgrade_from_series
        >>> ser = pd.Series(["John Doe","Jane Doe"],name="A",index=["e1","e2"])
        >>> ser
        e1    John Doe
        e2    Jane Doe
        Name: A, dtype: object
        >>> generic_upgrade_from_series(ser, reset_index=True)
           id    values
        0  e1  John Doe
        1  e2  Jane Doe
        Table Name: A, id_col: id
    """
    frame_class: Type[KlinkerFrame]
    id_col = "id"
    if isinstance(conc, pd.Series):
        frame_class = KlinkerPandasFrame
        if conc.index.name is None:
            conc.index.name = "id"
        else:
            id_col = conc.index.name
    else:
        frame_class = KlinkerDaskFrame
        if conc.index.name is None:
            conc._meta.index.name = "id"
        else:
            id_col = conc.index.name
    columns = ["values"] if not reset_index else [id_col, "values"]
    return frame_class._upgrade_from_series(
        conc,
        columns=columns,
        table_name=conc.name,
        id_col=id_col,
        reset_index=reset_index,
    )



if __name__ == "__main__":
    from sylloge import OAEI

    from klinker.data import KlinkerDataset

    ds = KlinkerDataset.from_sylloge(OAEI(backend="dask", npartitions=10))
    from klinker.blockers import TokenBlocker

    print(TokenBlocker().assign(left=ds.left, right=ds.right))

