import itertools
from typing import Dict, List, Tuple, Type

import dask.dataframe as dd
import pandas as pd
import pytest
from klinker.blockers import (
    DeepBlocker,
    EmbeddingBlocker,
    MinHashLSHBlocker,
    QgramsBlocker,
    StandardBlocker,
    TokenBlocker,
)
from klinker.blockers.base import Blocker
from klinker.blockers.relation_aware import (
    SimpleRelationalTokenBlocker,
    concat_neighbor_attributes,
)
from klinker.data import KlinkerBlockManager, KlinkerFrame, from_klinker_frame
from klinker.encoders.base import _get_ids
from mocks import MockKeyedVector
from strawman import dummy_triples
from util import assert_block_eq


@pytest.fixture()
def example_rel_triples() -> Tuple[pd.DataFrame, pd.DataFrame]:
    left_rel = dummy_triples(8, 5, entity_prefix="a")
    right_rel = dummy_triples(10, 8, entity_prefix="b")
    return left_rel, right_rel


@pytest.fixture()
def example_both(
    example_tables, example_triples
) -> Tuple[
    KlinkerFrame, KlinkerFrame, Tuple[str, str], Tuple[Dict[int, str], Dict[int, str]]
]:
    ta, _, dataset_names, id_mappings = example_tables
    _, tb, _, _ = example_triples
    return ta, tb, dataset_names, id_mappings


@pytest.fixture()
def example_prepostprocess() -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    return [
        pd.DataFrame({"A": {1: {"a2"}, 2: {"a1"}}, "B": {1: {"b2"}}}),
        pd.DataFrame({"A": {1: "a2", 2: "a1"}, "B": {1: {"b2"}}}),
        pd.DataFrame({"A": {1: "a2"}, "B": {1: {"b2"}}}),
    ], pd.DataFrame({"A": {1: {"a2"}}, "B": {1: {"b2"}}})


@pytest.fixture()
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


@pytest.fixture()
def expected_standard_blocker(example_tables) -> KlinkerBlockManager:
    _, _, dataset_names, id_mappings = example_tables
    return KlinkerBlockManager.from_dict(
        {"Bulgaria": (["a3"], ["b3"]), "USA": (["a1", "a2"], ["b2"])},
        dataset_names,
    )


@pytest.fixture()
def expected_qgrams_blocker(example_tables) -> pd.DataFrame:
    _, _, dataset_names, id_mappings = example_tables
    return KlinkerBlockManager.from_dict(
        {
            "Bul": (["a3"], ["b3"]),
            "Ind": (["a4"], ["b4"]),
            "USA": (["a1", "a2"], ["b2"]),
            "ari": (["a3"], ["b3"]),
            "gar": (["a3"], ["b3"]),
            "lga": (["a3"], ["b3"]),
            "ria": (["a3"], ["b3"]),
            "ulg": (["a3"], ["b3"]),
        },
        dataset_names,
    )


@pytest.fixture()
def expected_sorted_neighborhood_blocker(example_tables) -> KlinkerBlockManager:
    _, _, dataset_names, id_mappings = example_tables
    return KlinkerBlockManager.from_dict(
        {
            2: (["a3"], ["b3", "b4"]),
            3: (["a4"], ["b3", "b4"]),
            4: (["a4"], ["b5", "b3"]),
            5: (["a4", "a5"], ["b4"]),
            6: (["a5", "a1"], ["b4"]),
            8: (["a1", "a2"], ["b2"]),
            9: (["a2"], ["b2", "b0"]),
        },
        dataset_names,
    )


@pytest.fixture()
def expected_token_blocker(example_tables) -> KlinkerBlockManager:
    _, _, dataset_names, id_mappings = example_tables
    return KlinkerBlockManager.from_dict(
        {
            "02-02-1983": (["a2"], ["b2"]),
            "04-12-1990": (["a3"], ["b3", "b4"]),
            "11-12-1973": (["a1"], ["b1"]),
            "Bulgaria": (["a3"], ["b3"]),
            "John": (["a1"], ["b1"]),
            "Maggie": (["a2"], ["b2"]),
            "McExample": (["a1"], ["b1"]),
            "Nushi": (["a4"], ["b5"]),
            "Rebecca": (["a3"], ["b3"]),
            "Smith": (["a2", "a3"], ["b2", "b3"]),
            "USA": (["a1", "a2"], ["b2"]),
        },
        dataset_names,
    )


def assert_parquet(block: KlinkerBlockManager, tmp_dir):
    block.to_parquet(tmp_dir)
    block_pq = block.read_parquet(tmp_dir)
    assert_block_eq(block, block_pq)


@pytest.fixture()
def expected_lsh_blocker(example_tables) -> KlinkerBlockManager:
    _, _, dataset_names, id_mappings = example_tables
    return KlinkerBlockManager.from_dict(
        {0: (["a1"], ["b1"]), 1: (["a2"], ["b2"]), 2: (["a3"], ["b3"])},
        dataset_names,
    )


@pytest.mark.parametrize(
    "example_with_expected",
    list(
        itertools.product(
            ["example_tables"],
            [
                ("expected_standard_blocker", StandardBlocker, None),
                ("expected_qgrams_blocker", QgramsBlocker, None),
            ],
            [True, False],
            [False, True],
        )
    ),
    indirect=True,
)
def test_assign_schema_aware(example_with_expected, tmpdir):
    ta, tb, expected, cls = example_with_expected
    block = cls(blocking_key="BirthCountry").assign(ta, tb)
    assert_parquet(block, tmpdir)
    assert_block_eq(block, expected)


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
def test_assign_schema_agnostic(example_with_expected, tmpdir):
    ta, tb, expected, cls = example_with_expected
    block = cls().assign(ta, tb)
    assert_parquet(block, tmpdir)
    assert_block_eq(block, expected)


embedding_based_cases: List[Tuple] = list(
    itertools.product(
        [EmbeddingBlocker],
        ["AverageEmbeddingTokenizedFrameEncoder", "SIFEmbeddingTokenizedFrameEncoder"],
        [{}],
        [
            ("kiez", {"algorithm": "SklearnNN", "n_neighbors": 2}),
            ("hdbscan", {"min_cluster_size": 2}),
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
                    "kiez",
                    {"algorithm": "SklearnNN", "n_neighbors": 2},
                ),
                ("hdbscan", {"min_cluster_size": 2}),
            ],
        )
    )
)


@pytest.mark.parametrize(
    "tables", ["example_tables", "example_triples", "example_both"]
)
@pytest.mark.parametrize(
    ("cls", "frame_encoder", "frame_encoder_kwargs", "embedding_block_builder"),
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
    tmpdir,
):
    dimension = 3
    mock_kv_cls = MockKeyedVector
    mock_kv_cls.dimension = dimension
    mocker.patch(
        "klinker.encoders.pretrained.KeyedVectors",
        mock_kv_cls,
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
        save=False,
    )
    if use_dask and any(
        frame_encoder == noimp for noimp in ["CrossTupleTraining", "Hybrid"]
    ):
        with pytest.raises(NotImplementedError):
            block = blocker.assign(ta, tb)
    else:
        block = blocker.assign(ta, tb)
        assert_parquet(block, tmpdir)

        assert tuple(block.blocks.columns) == (ta.table_name, tb.table_name)
        if eb != "hdbscan":
            assert len(block) == len(
                ta.concat_values()
            )  # need unique in case of triples
            assert all(
                len(block_tuple[0]) == 1
                and len(block_tuple[1]) == eb_kwargs["n_neighbors"]
                for block_tuple in block.to_dict().values()
            )


@pytest.mark.parametrize(
    ("cls", "params"),
    [
        ("LightEAFrameEncoder", {"mini_dim": 3}),
        ("GCNFrameEncoder", {"layer_dims": 3, "use_weight_layers": True}),
        ("GCNFrameEncoder", {"layer_dims": 3, "use_weight_layers": False}),
    ],
)
def test_assign_relation_frame_encoder(
    cls,
    params,
    example_triples,
    example_rel_triples,
    mocker,
    tmpdir,
):
    dimension = 3
    mock_kv_cls = MockKeyedVector
    mock_kv_cls.dimension = dimension
    mocker.patch(
        "klinker.encoders.pretrained.KeyedVectors",
        mock_kv_cls,
    )
    ta, tb, _, _ = example_triples
    rel_ta, rel_tb = example_rel_triples
    eb_kwargs = {"algorithm": "SklearnNN", "n_neighbors": 2}
    block = EmbeddingBlocker(
        frame_encoder=cls,
        frame_encoder_kwargs=params,
        embedding_block_builder_kwargs=eb_kwargs,
        save=False,
    ).assign(ta, tb, rel_ta, rel_tb)

    assert_parquet(block, tmpdir)

    a_ids = _get_ids(attr=ta.set_index(ta.id_col), rel=rel_ta)
    assert tuple(block.blocks.columns) == (ta.table_name, tb.table_name)
    assert len(block) == len(a_ids)
    assert all(
        len(block_tuple[0]) == 1 and len(block_tuple[1]) == eb_kwargs["n_neighbors"]
        for block_tuple in block.to_dict().values()
    )


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("include_own_attributes", [True, False])
def test_concat_neighbor_attributes(
    example_tables, example_rel_triples, use_dask, include_own_attributes
):
    ta = example_tables[0]
    rel_ta = example_rel_triples[0]
    all_ids = set(rel_ta["head"].values).union(set(rel_ta["tail"].values))
    if include_own_attributes:
        all_ids |= set(ta["id"].values)

    if use_dask:
        ta = from_klinker_frame(ta, npartitions=2)
        rel_ta = dd.from_pandas(rel_ta, npartitions=2)
        conc_ta = concat_neighbor_attributes(
            ta, rel_ta, include_own_attributes=include_own_attributes
        ).compute()
    else:
        conc_ta = concat_neighbor_attributes(
            ta, rel_ta, include_own_attributes=include_own_attributes
        )
    assert len(conc_ta) == len(all_ids)
    assert set(conc_ta.index) == all_ids


@pytest.mark.parametrize("use_dask", [True, False])
def test_relational_token_blocker(
    example_tables, example_rel_triples, use_dask, tmpdir
):
    ta, tb, _, _ = example_tables
    rel_ta, rel_tb = example_rel_triples
    blocks = SimpleRelationalTokenBlocker().assign(ta, tb, rel_ta, rel_tb)
    assert_parquet(blocks, tmpdir)
    if use_dask:
        blocks.blocks.compute()
