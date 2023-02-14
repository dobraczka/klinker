from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from klinker.blockers import QgramsBlocker, StandardBlocker
from klinker.data import KlinkerFrame


@pytest.fixture
def example_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table_A = KlinkerFrame(
        df=pd.DataFrame(
            [
                [1, "John McExample", "11-12-1973", "USA", "Engineer"],
                [2, "Maggie Smith", "02-02-1983", "USA", "Scientist"],
                [3, "Rebecca Smith", "04-12-1990", "Bulgaria", "Chemist"],
                [4, "Nushi Devi", "14-03-1990", "India", None],
                [5, "Grzegorz Brzęczyszczykiewicz", "02-04-1970", "Poland", "Soldier"],
            ],
            columns=["id", "Name", "Birthdate", "BirthCountry", "Occupation"],
        ),
        name="A",
    )

    table_B = KlinkerFrame(
        df=pd.DataFrame(
            [
                [1, "John", "McExample", "11-12-1973", None],
                [2, "Maggie", "Smith", "02-02-1983", "USA"],
                [3, "Rebecca", "Smith", "04-12-1990", "Bulgaria"],
                [4, "Anh", "Nguyen", "04-12-1990", "Indonesia"],
                [5, "Nushi", "Zhang", "21-08-1989", "China"],
            ],
            columns=["id", "FirstName", "GivenName", "Birthdate", "BirthCountry"],
        ),
        name="B",
    )

    table_C = KlinkerFrame(
        df=pd.DataFrame(
            [
                [1, "Maximilian Müller", "13-12-1993", "Deutschland"],
                [2, "Grzegorz Brzęczyszczykiewicz", "02-04-1970", "Polska"],
                [3, "爱 李", "21-10-1989", "中华人民共和国"],
            ],
            columns=["id", "Name", "Birthdate", "BirthCountry"],
        ),
        name="C",
    )
    return table_A, table_B, table_C


@pytest.mark.parametrize(
    "cls, key, expected",
    [
        (
            StandardBlocker,
            "BirthCountry",
            {"A": {"Bulgaria": [3], "USA": [1, 2]}, "B": {"Bulgaria": [3], "USA": [2]}},
        ),
        (
            QgramsBlocker,
            "BirthCountry",
            {
                "A": {
                    "Bul": [2],
                    "Ind": [3],
                    "USA": [0, 1],
                    "ari": [2],
                    "gar": [2],
                    "lga": [2],
                    "ria": [2],
                    "ulg": [2],
                },
                "B": {
                    "Bul": [2],
                    "Ind": [3],
                    "USA": [1],
                    "ari": [2],
                    "gar": [2],
                    "lga": [2],
                    "ria": [2],
                    "ulg": [2],
                },
            },
        ),
    ],
)
def test_assign_schema_aware_binary(cls, key, expected, example_tables):
    ta, tb, _ = example_tables
    block = cls(blocking_key=key).assign([ta, tb])
    assert expected == block.to_dict()


@pytest.mark.parametrize(
    "cls, key, expected",
    [
        (
            StandardBlocker,
            "BirthCountry",
            {
                "A": {"Bulgaria": [3], "USA": [1, 2]},
                "B": {"Bulgaria": [3], "USA": [2]},
                "C": {"Bulgaria": np.nan, "USA": np.nan},
            },
        ),
        (
            QgramsBlocker,
            "BirthCountry",
            {
                "A": {
                    "Bul": [2],
                    "Ind": [3],
                    "Pol": [4],
                    "USA": [0, 1],
                    "and": [4],
                    "ari": [2],
                    "gar": [2],
                    "lan": [4],
                    "lga": [2],
                    "ria": [2],
                    "ulg": [2],
                },
                "B": {
                    "Bul": [2],
                    "Ind": [3],
                    "Pol": np.nan,
                    "USA": [1],
                    "and": np.nan,
                    "ari": [2],
                    "gar": [2],
                    "lan": np.nan,
                    "lga": [2],
                    "ria": [2],
                    "ulg": [2],
                },
                "C": {
                    "Bul": np.nan,
                    "Ind": np.nan,
                    "Pol": [1],
                    "USA": np.nan,
                    "and": [0],
                    "ari": np.nan,
                    "gar": np.nan,
                    "lan": [0],
                    "lga": np.nan,
                    "ria": np.nan,
                    "ulg": np.nan,
                },
            },
        ),
    ],
)
def test_assign_schema_aware_multi(cls, key, expected, example_tables):
    ta, tb, tc = example_tables
    block = cls(blocking_key=key).assign([ta, tb, tc])
    assert expected == block.to_dict()
