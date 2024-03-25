from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from klinker.blockers.embedding.blockbuilder import (
    HDBSCANEmbeddingBlockBuilder,
    KiezEmbeddingBlockBuilder,
)
from klinker.blockers.embedding.blocker import EmbeddingBlocker
from klinker.data import (
    KlinkerBlockManager,
    KlinkerFrame,
    KlinkerPandasFrame,
    NamedVector,
)
from strawman import dummy_triples
from util import assert_block_eq


def create_dummy_data(
    name: str, id_col: str = "head", length: int = 5, entity_prefix: str = "e"
) -> KlinkerFrame:
    return KlinkerPandasFrame.from_df(
        dummy_triples(
            length,
            num_entities=length,
            relation_triples=False,
            entity_prefix=entity_prefix,
        )
        .sort_values(by=id_col)
        .reindex(),
        table_name=name,
        id_col=id_col,
    )


@pytest.fixture()
def example() -> Tuple[NamedVector[np.ndarray], NamedVector[np.ndarray], str, str]:
    data = np.array(
        [
            [-10.02214045, 10.4762855],
            [4.40456117, -4.30098781],
            [-10.20858901, 9.92864014],
            [-8.41744546, 9.30333293],
            [-3.19137254, -6.58221354],
            [-4.36686114, -4.4813946],
            [6.07713967, -2.9225191],
            [-4.94871681, -3.99066116],
            [-9.6390726, 11.09024066],
            [4.37318904, -4.4791316],
        ]
    )
    left_length = 6
    right_length = len(data) - left_length
    left, right = data[:left_length], data[left_length:]
    left_names = [f"a{idx}" for idx in range(left_length)]
    right_names = [f"b{idx}" for idx in range(right_length)]
    return (
        NamedVector(names=left_names, vectors=left),
        NamedVector(names=right_names, vectors=right),
        "A",
        "B",
    )


@pytest.fixture()
def example_some_non_overlapping_clusters() -> (
    Tuple[NamedVector[np.ndarray], NamedVector[np.ndarray], str, str]
):
    np.random.seed(17)
    # create non-overlapping clusters
    left = np.random.normal(3, 1, size=(10, 4))
    right = np.random.normal(10, 1, size=(10, 4))

    # create common point
    left[5] = np.array([0.0, 0.0, 0.0, 0.0])
    right[5] = np.array([0.0, 0.0, 0.0, 0.0])

    left_names = [f"a{idx}" for idx in range(len(left))]
    right_names = [f"b{idx}" for idx in range(len(right))]
    return (
        NamedVector(names=left_names, vectors=left),
        NamedVector(names=right_names, vectors=right),
        "A",
        "B",
    )


@pytest.fixture()
def expected() -> KlinkerBlockManager:
    return KlinkerBlockManager.from_pandas(
        pd.DataFrame(
            {
                "A": {0: {"a0", "a2", "a3"}, 1: {"a4", "a5"}, 2: {"a1"}},
                "B": {0: {"b2"}, 1: {"b1"}, 2: {"b0", "b3"}},
            }
        )
    )


def test_cluster_block_builder(example, expected):
    blocks = HDBSCANEmbeddingBlockBuilder(min_cluster_size=2).build_blocks(*example)
    assert_block_eq(blocks, expected)


def test_nn_block_builder(example):
    blocks = KiezEmbeddingBlockBuilder(n_neighbors=2).build_blocks(*example)
    for btuple in blocks.to_dict().values():
        for ba in btuple[0]:
            assert ba.startswith("a")
        assert len(btuple[1]) == 2
        for bb in btuple[1]:
            assert bb.startswith("b")


def test_from_encoded(example, expected, tmp_path):
    left_enc, right_enc, table_name_a, table_name_b = example
    block_builder = HDBSCANEmbeddingBlockBuilder(min_cluster_size=2)
    mydir = tmp_path.joinpath("mysavedir")
    EmbeddingBlocker.save_encoded(
        mydir, (left_enc, right_enc), (table_name_a, table_name_b)
    )
    blocks = EmbeddingBlocker(
        embedding_block_builder=block_builder, save_dir=mydir
    ).from_encoded()
    assert_block_eq(blocks, expected)


def test_some_non_overlapping_clusters(example_some_non_overlapping_clusters):
    left_v, right_v, lname, rname = example_some_non_overlapping_clusters
    blocks = HDBSCANEmbeddingBlockBuilder(min_cluster_size=2).build_blocks(
        left=left_v, right=right_v, left_name=lname, right_name=rname
    )
    final_blocks = blocks.blocks.compute()
    assert len(final_blocks) == 1
    assert final_blocks["A"].values[0] == {"a5"}
    assert final_blocks["B"].values[0] == {"b5"}
