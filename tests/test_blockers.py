from typing import Tuple

import pandas as pd
import pytest
from mocks import MockGensimDownloader
from util import compare_blocks

from klinker.blockers import (
    DeepBlocker,
    EmbeddingBlocker,
    MinHashLSHBlocker,
    QgramsBlocker,
    SortedNeighborhoodBlocker,
    StandardBlocker,
    TokenBlocker,
)
from klinker.blockers.base import postprocess
from klinker.data import KlinkerFrame, KlinkerTripleFrame


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


@pytest.fixture
def example_triples(example_tables) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def triplify(df: pd.DataFrame) -> pd.DataFrame:
        new_df = (
            df.set_index("id")
            .apply(lambda row: [key for key, val in row.items()], axis=1)
            .explode()
            .to_frame()
            .rename(columns={df.name: "rel"})
        )
        new_df["tail"] = (
            df.set_index("id")
            .apply(lambda row: [val for key, val in row.items()], axis=1)
            .explode()
        )
        return KlinkerTripleFrame.from_df(
            new_df.reset_index(), name=df.name, id_col=df.id_col
        )

    table_A, table_B = example_tables
    return triplify(table_A), triplify(table_B)


@pytest.fixture
def example_both(request):
    ta, _ = request.getfixturevalue("example_tables")
    _, tb = request.getfixturevalue("example_triples")
    return ta, tb


@pytest.fixture
def example_prepostprocess() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return [
        pd.DataFrame({"A": {1: [2], 2: [1]}, "B": {1: [2]}}),
        pd.DataFrame({"A": {1: 2, 2: 1}, "B": {1: [2]}}),
        pd.DataFrame({"A": {1: 2}, "B": {1: [2]}}),
    ], pd.DataFrame({"A": {1: [2]}, "B": {1: [2]}})


@pytest.mark.parametrize(
    "cls, key, expected",
    [
        (
            StandardBlocker,
            "BirthCountry",
            pd.DataFrame(
                {
                    "A": {"Bulgaria": [3], "USA": [1, 2]},
                    "B": {"Bulgaria": [3], "USA": [2]},
                }
            ),
        ),
        (
            QgramsBlocker,
            "BirthCountry",
            pd.DataFrame(
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
                }
            ),
        ),
        (
            SortedNeighborhoodBlocker,
            "BirthCountry",
            pd.DataFrame(
                {
                    "A": {
                        2: [3],
                        3: [4],
                        4: [4],
                        5: [4, 5],
                        6: [5, 1],
                        8: [1, 2],
                        9: [2],
                    },
                    "B": {
                        2: [3, 5],
                        3: [3, 5],
                        4: [5, 4],
                        5: [4],
                        6: [4],
                        8: [2],
                        9: [2, 1],
                    },
                }
            ),
        ),
    ],
)
def test_assign_schema_aware(cls, key, expected, example_tables):
    ta, tb = example_tables
    block = cls(blocking_key=key).assign(ta, tb)
    compare_blocks(expected, block)


@pytest.mark.parametrize(
    "tables", ["example_tables", "example_triples", "example_both"]
)
@pytest.mark.parametrize(
    "cls, expected",
    [
        (
            TokenBlocker,
            pd.DataFrame(
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
                }
            ),
        ),
        (
            MinHashLSHBlocker,
            pd.DataFrame(
                {"A": {1: [1], 2: [2], 3: [3]}, "B": {1: [1], 2: [2], 3: [3]}}
            ),
        ),
    ],
)
def test_assign_schema_agnostic(tables, cls, expected, request):
    ta, tb = request.getfixturevalue(tables)
    block = cls().assign(ta, tb)
    compare_blocks(expected, block)


@pytest.mark.parametrize(
    "tables", ["example_tables", "example_triples", "example_both"]
)
@pytest.mark.parametrize(
    "expected",
    [
        pd.DataFrame(
            {
                "A": {0: [1], 1: [2], 2: [3], 3: [4], 4: [5]},
                "B": {0: [1, 2], 1: [2, 1], 2: [3, 2], 3: [4, 3], 4: [5, 3]},
            }
        ),
    ],
)
def test_assign_embedding_blocker(tables, expected, request, mocker):
    dimension = 3
    mocker.patch(
        "klinker.encoders.pretrained.gensim_downloader",
        MockGensimDownloader(dimension=dimension),
    )
    ta, tb = request.getfixturevalue(tables)
    block = EmbeddingBlocker(embedding_block_builder_kwargs=dict(n_neighbors=2)).assign(
        ta, tb
    )
    compare_blocks(expected, block)


def test_postprocess(example_prepostprocess):
    prepost, expected = example_prepostprocess
    for pp in prepost:
        assert postprocess(pp).equals(expected)


@pytest.mark.parametrize("encoder", ["AutoEncoder", "CrossTupleTraining", "Hybrid"])
def test_deep_blocker(encoder, example_tables, mocker):
    dimension = 3
    mocker.patch(
        "klinker.encoders.pretrained.gensim_downloader",
        MockGensimDownloader(dimension=dimension),
    )
    ta, tb = example_tables
    block = DeepBlocker(
        frame_encoder=encoder, frame_encoder_kwargs=dict(num_epochs=1)
    ).assign(ta, tb)
