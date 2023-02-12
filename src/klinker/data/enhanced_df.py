import numpy as np
import pandas as pd


@pd.api.extensions.register_dataframe_accessor("klinker")
class KlinkerAccessor:
    def __init__(self, pandas_obj):
        self._id_col = "id"
        self._validate(pandas_obj)
        self._obj = pandas_obj

    def _validate(self, obj):
        if obj.index.name != self._id_col:
            raise AttributeError(
                f"Must use {self._id_col} as index, but uses {obj.index.name}"
            )

    @property
    def id_col(self):
        return self._id_col

    @id_col.setter
    def id_col(self, value):
        self._id_col = value

    @property
    def ids(self):
        return self.prefix + self._obj[self.id_col].astype(str)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def suffix(self):
        return f"_{self._name}"


def klinkerfy(df: pd.DataFrame, name: str, id_col: str = "id") -> pd.DataFrame:
    df = df.set_index(id_col)
    df.klinker.name = name
    df.klinker.id_col = id_col
    return df


if __name__ == "__main__":

    table_A = pd.DataFrame(
        [
            [1, "John McExample", "11-12-1973", "USA", "Engineer"],
            [2, "Maggie Smith", "02-02-1983", "USA", "Scientist"],
            [3, "Rebecca Smith", "04-12-1990", "Bulgaria", "Chemist"],
            [4, "Nushi Devi", "14-03-1990", "India", None],
            [5, "Li Wang", "24-01-1995", "China", "Teacher"],
        ],
        columns=["id", "Name", "Birthdate", "BirthCountry", "Occupation"],
    )
    ta = klinkerfy(table_A, name="a")
    import ipdb  # noqa: autoimport

    ipdb.set_trace()  # BREAKPOINT
