from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from strawman import dummy_df

from klinker.data import KlinkerFrame


@pytest.fixture
def example() -> Dict:
    df = dummy_df((10, 3), columns=["colA", "colB", "colC"])
    return df.reset_index().to_dict()


@pytest.fixture
def block_example() -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    return (
        pd.DataFrame(
            {
                "A": {
                    2: [3],
                    4: [4],
                    5: [4],
                    6: [4, 5],
                    7: [5],
                    8: [5, 1],
                    9: [1, 2],
                    10: [1, 2],
                    11: [2],
                    3: np.nan,
                    12: np.nan,
                },
                "B": {
                    2: [3, 5],
                    4: [5],
                    5: [4],
                    6: [4],
                    7: [4],
                    8: np.nan,
                    9: np.nan,
                    10: [2],
                    11: [2],
                    3: [3, 5],
                    12: [2, 1],
                },
                "C": {
                    2: np.nan,
                    4: [1],
                    5: [1],
                    6: np.nan,
                    7: [2],
                    8: [2],
                    9: [2],
                    10: np.nan,
                    11: [3],
                    3: [1],
                    12: [3],
                },
            }
        ),
        {2: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 3: 3, 12: 3},
        3.0,
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


def test_klinkerframe_missing_id_col(example):
    with pytest.raises(AttributeError):
        KlinkerFrame(name="A", id_col="wrong", data=example)


def test_klinkerframe_non_unique_id_col():
    with pytest.raises(ValueError):
        KlinkerFrame(name="A", id_col="id", data={"id": {0: "id0", 1: "id1", 2: "id1"}})


def test_block_statistics(block_example):
    block, expected_sizes, mean_size = block_example
    assert expected_sizes == block.klinker_block.block_sizes.to_dict()
    assert mean_size == block.klinker_block.mean_block_size
