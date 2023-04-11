from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from strawman import dummy_triples
from util import compare_blocks

from klinker.blockers.embedding.blockbuilder import HDBSCANBlockBuilder
from klinker.data import KlinkerFrame
from klinker.typing import GeneralVector


def create_dummy_data(
    name: str, id_col: str = "head", length: int = 5, entity_prefix: str = "e"
) -> KlinkerFrame:
    return KlinkerFrame.from_df(
        dummy_triples(
            length,
            num_entities=length,
            relation_triples=False,
            entity_prefix=entity_prefix,
        )
        .sort_values(by=id_col)
        .reindex(),
        name=name,
        id_col=id_col,
    )


@pytest.fixture
def example() -> Tuple[GeneralVector, GeneralVector, KlinkerFrame, KlinkerFrame]:
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
    left_data = create_dummy_data("A", entity_prefix="a", length=left_length)
    right_data = create_dummy_data("B", entity_prefix="b", length=right_length)
    return left, right, left_data, right_data


@pytest.fixture
def expected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": {0: ["a0", "a2", "a3"], 1: ["a4", "a5"], 2: ["a1"]},
            "B": {0: ["b2"], 1: ["b1"], 2: ["b0", "b3"]},
        }
    )


def test_cluster_block_builder(example, expected):
    blocks = HDBSCANBlockBuilder(min_cluster_size=2).build_blocks(*example)
    compare_blocks(blocks, expected)
