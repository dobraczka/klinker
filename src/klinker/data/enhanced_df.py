from typing import Optional
import itertools

import pandas as pd
from pandas._typing import Axes, Dtype
from pandas.core.internals.base import DataManager


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

    def __repr__(self) -> str:
        return super().__repr__() + f"\nTable Name: {self.name}, id_col: {self.id_col}"


@pd.api.extensions.register_dataframe_accessor("klinker_block")
class BlockAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        pass

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
        tmp = self._obj.apply(lambda row: list(itertools.product(*row.tolist())), axis=1).explode()
        return pd.DataFrame(tmp.tolist(), index=tmp.index, columns=columns)
