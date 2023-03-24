from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from klinker.blockers import (
    MinHashLSHBlocker,
    QgramsBlocker,
    SortedNeighborhoodBlocker,
    StandardBlocker,
    TokenBlocker,
)
from klinker.data import KlinkerFrame


@pytest.fixture
def example_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    table_A = KlinkerFrame(
        data=[
            [1, "John McExample", "11-12-1973", "USA", "Engineer"],
            [2, "Maggie Smith", "02-02-1983", "USA", "Scientist"],
            [3, "Rebecca Smith", "04-12-1990", "Bulgaria", "Chemist"],
            [4, "Nushi Devi", "14-03-1990", "India", None],
            [5, "Grzegorz Brzęczyszczykiewicz", "02-04-1970", "Poland", "Soldier"],
        ],
        columns=["id", "Name", "Birthdate", "BirthCountry", "Occupation"],
        name="A",
    )

    table_B = KlinkerFrame(
        data=[
            [1, "John", "McExample", "11-12-1973", None],
            [2, "Maggie", "Smith", "02-02-1983", "USA"],
            [3, "Rebecca", "Smith", "04-12-1990", "Bulgaria"],
            [4, "Anh", "Nguyen", "04-12-1990", "Indonesia"],
            [5, "Nushi", "Zhang", "21-08-1989", "China"],
        ],
        name="B",
        columns=["id", "FirstName", "GivenName", "Birthdate", "BirthCountry"],
    )
    return table_A, table_B


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
        (
            SortedNeighborhoodBlocker,
            "BirthCountry",
            {
                "A": {2: [3], 3: [4], 4: [4], 5: [4, 5], 6: [5, 1], 8: [1, 2], 9: [2]},
                "B": {
                    2: [3, 5],
                    3: [3, 5],
                    4: [5, 4],
                    5: [4],
                    6: [4],
                    8: [2],
                    9: [2, 1],
                },
            },
        ),
    ],
)
def test_assign_schema_aware(cls, key, expected, example_tables):
    ta, tb = example_tables
    block = cls(blocking_key=key).assign(ta, tb)
    assert expected == block.to_dict()

@pytest.mark.parametrize(
    "cls, expected",
    [
        (
            TokenBlocker,
            {
                "A": {
                    "02-02-1983": [1],
                    "04-12-1990": [2],
                    "11-12-1973": [0],
                    "Bulgaria": [2],
                    "John": [0],
                    "Maggie": [1],
                    "McExample": [0],
                    "None": [3],
                    "Nushi": [3],
                    "Rebecca": [2],
                    "Smith": [1, 2],
                    "USA": [0, 1],
                },
                "B": {
                    "02-02-1983": [1],
                    "04-12-1990": [2, 3],
                    "11-12-1973": [0],
                    "Bulgaria": [2],
                    "John": [0],
                    "Maggie": [1],
                    "McExample": [0],
                    "None": [0],
                    "Nushi": [4],
                    "Rebecca": [2],
                    "Smith": [1, 2],
                    "USA": [1],
                },
            },
        ),
        (
            MinHashLSHBlocker,
            {"A": {1: [1], 2: [2], 3: [3]}, "B": {1: [1], 2: [2], 3: [3]}},
        ),
    ],
)
def test_assign_schema_agnostic(cls, expected, example_tables):
    ta, tb = example_tables
    block = cls().assign(ta, tb)
    assert expected == block.to_dict()
