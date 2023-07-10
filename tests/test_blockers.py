import itertools
from typing import Dict, List, Tuple, Type

import pandas as pd
import pytest
from mocks import MockGensimDownloader
from strawman import dummy_triples
import dask.dataframe as dd

from klinker.blockers import (
    DeepBlocker,
    EmbeddingBlocker,
    MinHashLSHBlocker,
    QgramsBlocker,
    SortedNeighborhoodBlocker,
    StandardBlocker,
    TokenBlocker,
)
from klinker.blockers.base import Blocker
from klinker.blockers.relation_aware import concat_neighbor_attributes
from klinker.data import KlinkerBlockManager, KlinkerFrame, KlinkerPandasFrame, KlinkerTriplePandasFrame, KlinkerDaskFrame, from_klinker_frame
from klinker.encoders.base import _get_ids


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


@pytest.fixture
def example_rel_triples() -> Tuple[pd.DataFrame, pd.DataFrame]:
    left_rel = dummy_triples(8, 5, entity_prefix="a")
    right_rel = dummy_triples(10, 8, entity_prefix="b")
    return left_rel, right_rel


@pytest.fixture
def example_both(
    example_tables, example_triples
) -> Tuple[
    KlinkerFrame, KlinkerFrame, Tuple[str, str], Tuple[Dict[int, str], Dict[int, str]]
]:
    ta, _, dataset_names, id_mappings = example_tables
    _, tb, _, _ = example_triples
    return ta, tb, dataset_names, id_mappings


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
    example, (expected, cls, index_prefix), stringify, use_dask = request.param
    ta, tb, _, _ = request.getfixturevalue(example)
    expected = request.getfixturevalue(expected)
    if use_dask:
        ta = from_klinker_frame(ta, npartitions=2)
        tb = from_klinker_frame(tb, npartitions=2)
    return ta, tb, expected, cls


@pytest.fixture
def expected_standard_blocker(example_tables) -> KlinkerBlockManager:
    _, _, dataset_names, id_mappings = example_tables
    return KlinkerBlockManager(
        {"Bulgaria": ({2}, {2}), "USA": ({0, 1}, {1})},
        dataset_names,
        id_mappings,
    )


@pytest.fixture
def expected_qgrams_blocker(example_tables) -> pd.DataFrame:
    _, _, dataset_names, id_mappings = example_tables
    return KlinkerBlockManager(
        {
            "Bul": ({2}, {2}),
            "Ind": ({3}, {3}),
            "USA": ({0, 1}, {1}),
            "ari": ({2}, {2}),
            "gar": ({2}, {2}),
            "lga": ({2}, {2}),
            "ria": ({2}, {2}),
            "ulg": ({2}, {2}),
        },
        dataset_names,
        id_mappings,
    )


@pytest.fixture
def expected_sorted_neighborhood_blocker(example_tables) -> KlinkerBlockManager:
    _, _, dataset_names, id_mappings = example_tables
    return KlinkerBlockManager(
        {
            2: ({2}, {2, 4}),
            3: ({3}, {2, 4}),
            4: ({3}, {4, 3}),
            5: ({3, 4}, {3}),
            6: ({4, 0}, {3}),
            8: ({0, 1}, {1}),
            9: ({1}, {1, 0}),
        },
        dataset_names,
        id_mappings,
    )


@pytest.fixture
def expected_token_blocker(example_tables) -> KlinkerBlockManager:
    _, _, dataset_names, id_mappings = example_tables
    return KlinkerBlockManager(
        {
            "02-02-1983": ({1}, {1}),
            "04-12-1990": ({2}, {2, 3}),
            "11-12-1973": ({0}, {0}),
            "Bulgaria": ({2}, {2}),
            "John": ({0}, {0}),
            "Maggie": ({1}, {1}),
            "McExample": ({0}, {0}),
            "None": ({3}, {0}),
            "Nushi": ({3}, {4}),
            "Rebecca": ({2}, {2}),
            "Smith": ({1, 2}, {1, 2}),
            "USA": ({0, 1}, {1}),
        },
        dataset_names,
        id_mappings,
    )


@pytest.fixture
def expected_lsh_blocker(example_tables) -> KlinkerBlockManager:
    _, _, dataset_names, id_mappings = example_tables
    return KlinkerBlockManager(
        {1: ({0}, {0}), 2: ({1}, {1}), 3: ({2}, {2})}, dataset_names, id_mappings
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
            [False, True],
        )
    ),
    indirect=True,
)
def test_assign_schema_aware(example_with_expected):
    ta, tb, expected, cls = example_with_expected
    if cls == SortedNeighborhoodBlocker and isinstance(ta, KlinkerDaskFrame):
        with pytest.raises(ValueError):
            cls(blocking_key="BirthCountry").assign(ta, tb)
    else:
        block = cls(blocking_key="BirthCountry").assign(ta, tb)
        block == expected


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
            [False, True],
        )
    ),
    indirect=True,
)
def test_assign_schema_agnostic(example_with_expected):
    ta, tb, expected, cls = example_with_expected
    block = cls().assign(ta, tb)
    block == expected


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
@pytest.mark.parametrize("use_dask", [True, False])
def test_assign_embedding_blocker(
    tables,
    cls,
    frame_encoder,
    frame_encoder_kwargs,
    embedding_block_builder,
    use_dask,
    request,
    mocker,
):
    dimension = 3
    mocker.patch(
        "klinker.encoders.pretrained.gensim_downloader",
        MockGensimDownloader(dimension=dimension),
    )
    ta, tb, _, _ = request.getfixturevalue(tables)
    if use_dask:
        ta = from_klinker_frame(ta, npartitions=2)
        tb = from_klinker_frame(tb, npartitions=2)

    eb, eb_kwargs = embedding_block_builder
    blocker = cls(
        frame_encoder=frame_encoder,
        frame_encoder_kwargs=frame_encoder_kwargs,
        embedding_block_builder=eb,
        embedding_block_builder_kwargs=eb_kwargs,
    )
    if use_dask and any([frame_encoder == noimp for noimp in ["CrossTupleTraining", "Hybrid"]]):
        with pytest.raises(NotImplementedError):
            block = blocker.assign(ta, tb)
    else:
        block = blocker.assign(ta, tb)

        assert block.dataset_names == (ta.table_name, tb.table_name)
        if eb != "HDBSCANBlockBuilder":
            assert len(block) == len(ta.concat_values())  # need unique in case of triples
            assert all(
                len(block_tuple[0]) == 1 and len(block_tuple[1]) == eb_kwargs["n_neighbors"]
                for block_tuple in block.values()
            )


@pytest.mark.parametrize(
    "cls, params", [("LightEAFrameEncoder", dict(mini_dim=3)), ("GCNFrameEncoder", {})]
)
def test_assign_relation_frame_encoder(
    cls,
    params,
    example_triples,
    example_rel_triples,
    mocker,
):
    dimension = 3
    mocker.patch(
        "klinker.encoders.pretrained.gensim_downloader",
        MockGensimDownloader(dimension=dimension),
    )
    ta, tb, _, _ = example_triples
    rel_ta, rel_tb = example_rel_triples
    eb_kwargs = {"algorithm": "SklearnNN", "n_neighbors": 2}
    block = EmbeddingBlocker(
        frame_encoder=cls,
        frame_encoder_kwargs=params,
        embedding_block_builder_kwargs=eb_kwargs,
    ).assign(ta, tb, rel_ta, rel_tb)

    a_ids = _get_ids(attr=ta.set_index(ta.id_col), rel=rel_ta)
    assert block.dataset_names == (ta.table_name, tb.table_name)
    assert len(block) == len(a_ids)
    assert all(
        len(block_tuple[0]) == 1 and len(block_tuple[1]) == eb_kwargs["n_neighbors"]
        for block_tuple in block.values()
    )


# def test_postprocess(example_prepostprocess):
#     prepost, expected = example_prepostprocess
#     for pp in prepost:
#         assert postprocess(pp).equals(expected)


@pytest.mark.parametrize("use_dask", [True, False])
def test_concat_neighbor_attributes(example_tables, example_rel_triples, use_dask):
    ta = example_tables[0]
    rel_ta = example_rel_triples[0]
    all_ids = set(rel_ta["head"].values).union(set(rel_ta["tail"].values))

    if use_dask:
        ta = from_klinker_frame(ta, npartitions=2)
        rel_ta = dd.from_pandas(rel_ta, npartitions=2)
        conc_ta = concat_neighbor_attributes(ta, rel_ta).compute()
    else:
        conc_ta = concat_neighbor_attributes(ta, rel_ta)
    assert len(conc_ta) == len(all_ids)
