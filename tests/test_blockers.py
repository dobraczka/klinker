import itertools
from typing import List, Optional, Tuple, Type

import pandas as pd
import pytest
from mocks import MockGensimDownloader
from strawman import dummy_triples
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
from klinker.blockers.base import Blocker, postprocess
from klinker.data import KlinkerFrame, KlinkerTripleFrame


def stringify_example_id(
    ta: KlinkerFrame, tb: KlinkerFrame
) -> Tuple[KlinkerFrame, KlinkerFrame]:
    ta["id"] = "a" + ta["id"].astype(str)
    tb["id"] = "b" + tb["id"].astype(str)
    return ta, tb


def stringify_expected_id(
    expected: pd.DataFrame, index_prefix: Optional[str] = None
) -> pd.DataFrame:
    expected["A"] = expected["A"].apply(lambda x: ["a" + str(xi) for xi in x])
    expected["B"] = expected["B"].apply(lambda x: ["b" + str(xi) for xi in x])
    if index_prefix is not None:
        expected.index = index_prefix + expected.index.astype(str)
    return expected


@pytest.fixture
def example_tables() -> Tuple[KlinkerFrame, KlinkerFrame]:
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
            [10, "John", "McExample", "11-12-1973", None],
            [20, "Maggie", "Smith", "02-02-1983", "USA"],
            [30, "Rebecca", "Smith", "04-12-1990", "Bulgaria"],
            [40, "Anh", "Nguyen", "04-12-1990", "Indonesia"],
            [50, "Nushi", "Zhang", "21-08-1989", "China"],
        ],
        name="B",
        columns=["id", "FirstName", "GivenName", "Birthdate", "BirthCountry"],
    )
    return table_A, table_B


@pytest.fixture
def example_triples(example_tables) -> Tuple[KlinkerFrame, KlinkerFrame]:
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
def example_rel_triples() -> Tuple[pd.DataFrame, pd.DataFrame]:
    left_rel = dummy_triples(8, 5, entity_prefix="")
    right_rel = dummy_triples(10, 8, entity_prefix="")
    return left_rel, right_rel


@pytest.fixture
def example_both(example_tables, example_triples) -> Tuple[KlinkerFrame, KlinkerFrame]:
    ta, _ = example_tables
    _, tb = example_triples
    return ta, tb


@pytest.fixture
def example_prepostprocess() -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    return [
        pd.DataFrame({"A": {1: [2], 2: [1]}, "B": {1: [20]}}),
        pd.DataFrame({"A": {1: 2, 2: 1}, "B": {1: [20]}}),
        pd.DataFrame({"A": {1: 2}, "B": {1: [20]}}),
    ], pd.DataFrame({"A": {1: [2]}, "B": {1: [20]}})


@pytest.fixture
def example_with_expected(
    request,
) -> Tuple[KlinkerFrame, KlinkerFrame, pd.DataFrame, Type[Blocker]]:
    example, (expected, cls, index_prefix), stringify = request.param
    ta, tb = request.getfixturevalue(example)
    expected = request.getfixturevalue(expected)
    if stringify:
        ta, tb = stringify_example_id(ta, tb)
        expected = stringify_expected_id(expected, index_prefix)
    return ta, tb, expected, cls


@pytest.fixture
def expected_standard_blocker() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": {"Bulgaria": [3], "USA": [1, 2]},
            "B": {"Bulgaria": [30], "USA": [20]},
        }
    )


@pytest.fixture
def expected_qgrams_blocker() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": {
                "Bul": [3],
                "Ind": [4],
                "USA": [1, 2],
                "ari": [3],
                "gar": [3],
                "lga": [3],
                "ria": [3],
                "ulg": [3],
            },
            "B": {
                "Bul": [30],
                "Ind": [40],
                "USA": [20],
                "ari": [30],
                "gar": [30],
                "lga": [30],
                "ria": [30],
                "ulg": [30],
            },
        }
    )


@pytest.fixture
def expected_sorted_neighborhood_blocker() -> pd.DataFrame:
    return pd.DataFrame(
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
                2: [30, 50],
                3: [30, 50],
                4: [50, 40],
                5: [40],
                6: [40],
                8: [20],
                9: [20, 10],
            },
        }
    )


@pytest.fixture
def expected_token_blocker() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": {
                "02-02-1983": [2],
                "04-12-1990": [3],
                "11-12-1973": [1],
                "Bulgaria": [3],
                "John": [1],
                "Maggie": [2],
                "McExample": [1],
                "None": [4],
                "Nushi": [4],
                "Rebecca": [3],
                "Smith": [2, 3],
                "USA": [1, 2],
            },
            "B": {
                "02-02-1983": [20],
                "04-12-1990": [30, 40],
                "11-12-1973": [10],
                "Bulgaria": [30],
                "John": [10],
                "Maggie": [20],
                "McExample": [10],
                "None": [10],
                "Nushi": [50],
                "Rebecca": [30],
                "Smith": [20, 30],
                "USA": [20],
            },
        }
    )


@pytest.fixture
def expected_lsh_blocker() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": {1: [1], 2: [2], 3: [3]}, "B": {1: [10], 2: [20], 3: [30]}}
    )


@pytest.mark.parametrize(
    "example_with_expected",
    list(
        itertools.product(
            ["example_tables"],
            [
                ("expected_standard_blocker", StandardBlocker, None),
                ("expected_qgrams_blocker", QgramsBlocker, None),
                (
                    "expected_sorted_neighborhood_blocker",
                    SortedNeighborhoodBlocker,
                    None,
                ),
            ],
            [True, False],
        )
    ),
    indirect=True,
)
def test_assign_schema_aware(example_with_expected):
    ta, tb, expected, cls = example_with_expected
    block = cls(blocking_key="BirthCountry").assign(ta, tb)
    compare_blocks(expected, block)


@pytest.mark.parametrize(
    "example_with_expected",
    list(
        itertools.product(
            ["example_tables", "example_triples", "example_both"],
            [
                ("expected_token_blocker", TokenBlocker, None),
                ("expected_lsh_blocker", MinHashLSHBlocker, "b"),
            ],
            [True, False],
        )
    ),
    indirect=True,
)
def test_assign_schema_agnostic(example_with_expected):
    ta, tb, expected, cls = example_with_expected
    block = cls().assign(ta, tb)
    compare_blocks(expected, block)


embedding_based_cases: List[Tuple] = list(
    itertools.product(
        [EmbeddingBlocker],
        ["AverageEmbeddingTokenizedFrameEncoder", "SIFEmbeddingTokenizedFrameEncoder"],
        [{}],
        [
            ("KiezEmbeddingBlockBuilder", {"algorithm": "SklearnNN", "n_neighbors": 2}),
            ("HDBSCANBlockBuilder", {"min_cluster_size": 2}),
        ],
    )
)

embedding_based_cases.extend(
    list(
        itertools.product(
            [DeepBlocker],
            ["AutoEncoder", "CrossTupleTraining", "Hybrid"],
            [{"num_epochs": 1}],
            [
                (
                    "KiezEmbeddingBlockBuilder",
                    {"algorithm": "SklearnNN", "n_neighbors": 2},
                ),
                ("HDBSCANBlockBuilder", {"min_cluster_size": 2}),
            ],
        )
    )
)


@pytest.mark.parametrize(
    "tables", ["example_tables", "example_triples", "example_both"]
)
@pytest.mark.parametrize(
    "cls, frame_encoder, frame_encoder_kwargs, embedding_block_builder",
    embedding_based_cases,
)
def test_assign_embedding_blocker(
    tables,
    cls,
    frame_encoder,
    frame_encoder_kwargs,
    embedding_block_builder,
    request,
    mocker,
):
    dimension = 3
    mocker.patch(
        "klinker.encoders.pretrained.gensim_downloader",
        MockGensimDownloader(dimension=dimension),
    )
    ta, tb = request.getfixturevalue(tables)
    eb, eb_kwargs = embedding_block_builder
    block = cls(
        frame_encoder=frame_encoder,
        frame_encoder_kwargs=frame_encoder_kwargs,
        embedding_block_builder=eb,
        embedding_block_builder_kwargs=eb_kwargs,
    ).assign(ta, tb)

    assert block.columns.tolist() == [ta.name, tb.name]
    if eb != "HDBSCANBlockBuilder":
        assert len(block) == len(ta.concat_values())  # need unique in case of triples
        assert all(len(val) == 1 for val in block[ta.name].values)
        assert all(
            len(val) == eb_kwargs["n_neighbors"] for val in block[tb.name].values
        )


def test_assign_light_ea(
    example_triples,
    example_rel_triples,
    mocker,
):
    dimension = 3
    mocker.patch(
        "klinker.encoders.pretrained.gensim_downloader",
        MockGensimDownloader(dimension=dimension),
    )
    ta, tb = example_triples
    rel_ta, rel_tb = example_rel_triples
    eb_kwargs = {"algorithm": "SklearnNN", "n_neighbors": 2}
    block = EmbeddingBlocker(
        frame_encoder="LightEAFrameEncoder",
    ).assign(ta, tb, rel_ta, rel_tb)

    assert block.columns.tolist() == [ta.name, tb.name]
    assert len(block) == len(ta.concat_values())  # need unique in case of triples
    assert all(len(val) == 1 for val in block[ta.name].values)
    assert all(len(val) == eb_kwargs["n_neighbors"] for val in block[tb.name].values)


def test_postprocess(example_prepostprocess):
    prepost, expected = example_prepostprocess
    for pp in prepost:
        assert postprocess(pp).equals(expected)
