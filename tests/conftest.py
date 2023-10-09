from typing import Dict, Tuple

import pandas as pd
import pytest

from klinker.data import KlinkerFrame, KlinkerPandasFrame, KlinkerTriplePandasFrame


@pytest.fixture
def example_tables() -> Tuple[
    KlinkerFrame, KlinkerFrame, Tuple[str, str], Tuple[Dict[int, str], Dict[int, str]]
]:
    dataset_names = ("A", "B")
    id_mappings = (
        {id_num: f"a{id_num}" for id_num in range(0, 5)},
        {id_num: f"b{id_num}" for id_num in range(0, 5)},
    )
    table_A = KlinkerPandasFrame(
        data=[
            ["a1", "John McExample", "11-12-1973", "USA", "Engineer"],
            ["a2", "Maggie Smith", "02-02-1983", "USA", "Scientist"],
            ["a3", "Rebecca Smith", "04-12-1990", "Bulgaria", "Chemist"],
            ["a4", "Nushi Devi", "14-03-1990", "India", None],
            ["a5", "Grzegorz Brzęczyszczykiewicz", "02-04-1970", "Poland", "Soldier"],
        ],
        columns=["id", "Name", "Birthdate", "BirthCountry", "Occupation"],
        table_name=dataset_names[0],
    )

    table_B = KlinkerPandasFrame(
        data=[
            ["b1", "John", "McExample", "11-12-1973", None],
            ["b2", "Maggie", "Smith", "02-02-1983", "USA"],
            ["b3", "Rebecca", "Smith", "04-12-1990", "Bulgaria"],
            ["b4", "Anh", "Nguyen", "04-12-1990", "Indonesia"],
            ["b5", "Nushi", "Zhang", "21-08-1989", "China"],
        ],
        columns=["id", "FirstName", "GivenName", "Birthdate", "BirthCountry"],
        table_name=dataset_names[1],
    )
    return table_A, table_B, dataset_names, id_mappings


@pytest.fixture
def example_triples(
    example_tables,
) -> Tuple[
    KlinkerFrame, KlinkerFrame, Tuple[str, str], Tuple[Dict[int, str], Dict[int, str]]
]:
    def triplify(df: pd.DataFrame) -> pd.DataFrame:
        new_df = (
            df.set_index("id")
            .apply(lambda row: [key for key, val in row.items()], axis=1)
            .explode()
            .to_frame()
            .rename(columns={df.table_name: "rel"})
        )
        new_df["tail"] = (
            df.set_index("id")
            .apply(lambda row: [val for key, val in row.items()], axis=1)
            .explode()
        )
        return KlinkerTriplePandasFrame.from_df(
            new_df.reset_index(), table_name=df.table_name, id_col=df.id_col
        )

    table_A, table_B, dataset_names, id_mappings = example_tables
    return triplify(table_A), triplify(table_B), dataset_names, id_mappings
