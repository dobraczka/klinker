from typing import Dict, Tuple

import pandas as pd
import pytest
from strawman import dummy_df, dummy_triples

from klinker.data import KlinkerFrame, KlinkerTripleFrame


@pytest.fixture
def example() -> Dict:
    df = dummy_df((10, 3), columns=["colA", "colB", "colC"])
    return df.reset_index().to_dict()


@pytest.fixture
def triple_example() -> Dict:
    return dummy_triples(10, relation_triples=False).to_dict()


@pytest.fixture
def block_example() -> Tuple[
    pd.DataFrame, Dict[int, int], float, Dict[str, Dict[int, int]]
]:
    return (
        pd.DataFrame(
            {
                "A": {
                    2: [3],
                    4: [4],
                    5: [4],
                    6: [4, 5],
                    7: [5],
                    10: [1, 2],
                    11: [2],
                },
                "B": {
                    2: [3, 5],
                    4: [5],
                    5: [4],
                    6: [4],
                    7: [4],
                    10: [2],
                    11: [2],
                },
            }
        ),
        {2: 3, 4: 2, 5: 2, 6: 3, 7: 2, 10: 3, 11: 2},
        2.4285714285714284,
        {
            "A": {2: 3, 4: 4, 5: 4, 6: 5, 7: 5, 10: 2, 11: 2},
            "B": {2: 5, 4: 5, 5: 4, 6: 4, 7: 4, 10: 2, 11: 2},
        },
    )


def test_klinkerframe(example):
    name = "A"
    id_col = "index"
    kf = KlinkerFrame(name=name, id_col=id_col, data=example)
    assert kf.name == name
    assert kf.id_col == id_col

    # check that metadata is present after transformation
    transformed = kf[[kf.id_col, kf.columns[1], kf.columns[2]]]
    assert not transformed.empty
    assert transformed.name == name
    assert transformed.id_col == id_col

    # check prefixed
    assert kf.prefixed_ids.to_list() == [name + "_" + str(i) for i in range(len(kf))]

    # check non-id-columns
    assert kf.non_id_columns == ["colA", "colB", "colC"]


def test_klinker_triple_frame(triple_example):
    name = "A"
    id_col = "head"
    ktf = KlinkerTripleFrame(data=triple_example, name=name, id_col=id_col)
    assert ktf.non_id_columns == ["tail"]
    concated = ktf.concat_values()
    assert len(concated[concated.id_col].unique()) == len(concated)


def test_block_statistics(block_example):
    block, expected_sizes, mean_size, _ = block_example
    assert expected_sizes == block.klinker_block.block_sizes.to_dict()
    assert mean_size == pytest.approx(block.klinker_block.mean_block_size)


def test_block_to_pairs(block_example):
    block, _, _, pairs = block_example
    assert pairs == block.klinker_block.to_pairs().to_dict()


def test_validate_klinker_block():
    with pytest.raises(ValueError):
        # only lists allowed
        pd.DataFrame({"A": {1: 1, 2: [1]}, "B": {1: [1], 2: [1]}}).klinker_block
    with pytest.raises(ValueError):
        # no nans
        pd.DataFrame({"A": {4: [1], 2: [1]}, "B": {1: [1], 2: [1]}}).klinker_block
