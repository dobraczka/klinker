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
from klinker.encoders.base import _get_ids
from klinker.data import KlinkerFrame, KlinkerTripleFrame

@pytest.fixture
def example_tables() -> Tuple[KlinkerFrame, KlinkerFrame]:
    table_A = KlinkerFrame(
        data=[
            ["a1", "John McExample", "11-12-1973", "USA", "Engineer"],
            ["a2", "Maggie Smith", "02-02-1983", "USA", "Scientist"],
            ["a3", "Rebecca Smith", "04-12-1990", "Bulgaria", "Chemist"],
            ["a4", "Nushi Devi", "14-03-1990", "India", None],
            ["a5", "Grzegorz Brzęczyszczykiewicz", "02-04-1970", "Poland", "Soldier"],
        ],
        columns=["id", "Name", "Birthdate", "BirthCountry", "Occupation"],
        name="A",
    )

    table_B = KlinkerFrame(
        data=[
            ["b1", "John", "McExample", "11-12-1973", None],
            ["b2", "Maggie", "Smith", "02-02-1983", "USA"],
            ["b3", "Rebecca", "Smith", "04-12-1990", "Bulgaria"],
            ["b4", "Anh", "Nguyen", "04-12-1990", "Indonesia"],
            ["b5", "Nushi", "Zhang", "21-08-1989", "China"],
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
    left_rel = dummy_triples(8, 5, entity_prefix="a")
    right_rel = dummy_triples(10, 8, entity_prefix="b")
    return left_rel, right_rel


@pytest.fixture
def example_both(example_tables, example_triples) -> Tuple[KlinkerFrame, KlinkerFrame]:
    ta, _ = example_tables
    _, tb = example_triples
    return ta, tb


@pytest.fixture
def example_prepostprocess() -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    return [
        pd.DataFrame({"A": {1: {"a2"}, 2: {"a1"}}, "B": {1: {"b2"}}}),
        pd.DataFrame({"A": {1: "a2", 2: "a1"}, "B": {1: {"b2"}}}),
        pd.DataFrame({"A": {1: "a2"}, "B": {1: {"b2"}}}),
    ], pd.DataFrame({"A": {1: {"a2"}}, "B": {1: {"b2"}}})


@pytest.fixture
def example_with_expected(
    request,
) -> Tuple[KlinkerFrame, KlinkerFrame, pd.DataFrame, Type[Blocker]]:
    example, (expected, cls, index_prefix), stringify = request.param
    ta, tb = request.getfixturevalue(example)
    expected = request.getfixturevalue(expected)
    return ta, tb, expected, cls


@pytest.fixture
def expected_standard_blocker() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": {"Bulgaria": {"a3"}, "USA": {"a1", "a2"}},
            "B": {"Bulgaria": {"b3"}, "USA": {"b2"}},
        }
    )


@pytest.fixture
def expected_qgrams_blocker() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": {
                "Bul": {"a3"},
                "Ind": {"a4"},
                "USA": {"a1", "a2"},
                "ari": {"a3"},
                "gar": {"a3"},
                "lga": {"a3"},
                "ria": {"a3"},
                "ulg": {"a3"},
            },
            "B": {
                "Bul": {"b3"},
                "Ind": {"b4"},
                "USA": {"b2"},
                "ari": {"b3"},
                "gar": {"b3"},
                "lga": {"b3"},
                "ria": {"b3"},
                "ulg": {"b3"},
            },
        }
    )


@pytest.fixture
def expected_sorted_neighborhood_blocker() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": {
                2: {"a3"},
                3: {"a4"},
                4: {"a4"},
                5: {"a4", "a5"},
                6: {"a5", "a1"},
                8: {"a1", "a2"},
                9: {"a2"},
            },
            "B": {
                2: {"b3", "b5"},
                3: {"b3", "b5"},
                4: {"b5", "b4"},
                5: {"b4"},
                6: {"b4"},
                8: {"b2"},
                9: {"b2", "b1"},
            },
        }
    )


@pytest.fixture
def expected_token_blocker() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": {
                "02-02-1983": {"a2"},
                "04-12-1990": {"a3"},
                "11-12-1973": {"a1"},
                "Bulgaria": {"a3"},
                "John": {"a1"},
                "Maggie": {"a2"},
                "McExample": {"a1"},
                "None": {"a4"},
                "Nushi": {"a4"},
                "Rebecca": {"a3"},
                "Smith": {"a2", "a3"},
                "USA": {"a1", "a2"},
            },
            "B": {
                "02-02-1983": {"b2"},
                "04-12-1990": {"b3", "b4"},
                "11-12-1973": {"b1"},
                "Bulgaria": {"b3"},
                "John": {"b1"},
                "Maggie": {"b2"},
                "McExample": {"b1"},
                "None": {"b1"},
                "Nushi": {"b5"},
                "Rebecca": {"b3"},
                "Smith": {"b2", "b3"},
                "USA": {"b2"},
            },
        }
    )


@pytest.fixture
def expected_lsh_blocker() -> pd.DataFrame:
    return pd.DataFrame(
        {"A": {1: {"a1"}, 2: {"a2"}, 3: {"a3"}}, "B": {1: {"b1"}, 2: {"b2"}, 3: {"b3"}}}
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
        frame_encoder_kwargs=dict(mini_dim=3),
        embedding_block_builder_kwargs=eb_kwargs,
    ).assign(ta, tb, rel_ta, rel_tb)

    a_ids = _get_ids(attr=ta.set_index(ta.id_col), rel=rel_ta)
    assert block.columns.tolist() == [ta.name, tb.name]
    assert len(block) == len(a_ids)
    assert all(len(val) == 1 for val in block[ta.name].values)
    assert all(len(val) == eb_kwargs["n_neighbors"] for val in block[tb.name].values)


def test_postprocess(example_prepostprocess):
    prepost, expected = example_prepostprocess
    for pp in prepost:
        assert postprocess(pp).equals(expected)
