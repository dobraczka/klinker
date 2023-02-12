from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from klinker.blockers import StandardBlocker
from klinker.data import klinkerfy


@pytest.fixture
def example_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table_A = klinkerfy(
        pd.DataFrame(
            [
                [1, "John McExample", "11-12-1973", "USA", "Engineer"],
                [2, "Maggie Smith", "02-02-1983", "USA", "Scientist"],
                [3, "Rebecca Smith", "04-12-1990", "Bulgaria", "Chemist"],
                [4, "Nushi Devi", "14-03-1990", "India", None],
                [5, "Li Wang", "24-01-1995", "China", "Teacher"],
            ],
            columns=["id", "Name", "Birthdate", "BirthCountry", "Occupation"],
        ),
        name="A",
    )

    table_B = klinkerfy(
        pd.DataFrame(
            [
                [1, "John", "McExample", "11-12-1973", "Germany"],
                [2, "Maggie", "Smith", "02-02-1983", "USA"],
                [3, "Rebecca", "Smith", "04-12-1990", "Bulgaria"],
                [4, "Anh", "Nguyen", "04-12-1990", "Indonesia"],
                [5, "Nushi", "Zhang", "21-08-1989", "China"],
            ],
            columns=["id", "FirstName", "GivenName", "Birthdate", "BirthCountry"],
        ),
        name="B",
    )

    table_C = klinkerfy(
        pd.DataFrame(
            [
                [1, "Maximilian Müller", "13-12-1993", "Germany"],
                [2, "Grzegorz Brzęczyszczykiewicz", "02-04-1970", "Poland"],
                [3, "爱 李", "21-10-1989", "China"],
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
            {
                "A": {"Bulgaria": [3], "China": [5], "USA": [1, 2]},
                "B": {"Bulgaria": [3], "China": [5], "USA": [2]},
            },
        )
    ],
)
def test_assign_schema_aware_binary(cls, key, expected, example_tables):
    ta, tb,_ = example_tables
    block = StandardBlocker(blocking_key=key).assign([ta, tb])
    assert expected == block.to_dict()


@pytest.mark.parametrize(
    "cls, key, expected",
    [
        (
            StandardBlocker,
            "BirthCountry",
            {
                "A": {"Bulgaria": [3], "China": [5], "USA": [1, 2], "Germany": np.nan},
                "B": {"Bulgaria": [3], "China": [5], "USA": [2], "Germany": [1]},
                "C": {"Bulgaria": np.nan, "China": [3], "USA": np.nan, "Germany":[1]},
            },
        )
    ],
)
def test_assign_schema_aware_multi(cls, key, expected, example_tables):
    ta, tb, tc = example_tables
    block = StandardBlocker(blocking_key=key).assign([ta, tb, tc])
    assert expected == block.to_dict()
