from typing import Dict

import pytest
from strawman import dummy_df

from klinker.data import KlinkerFrame


@pytest.fixture
def example() -> Dict:
    df = dummy_df((10, 3), columns=["colA", "colB", "colC"])
    return df.reset_index().to_dict()


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
        KlinkerFrame(name="A", id_col="id", data={"id":{0:"id0", 1:"id1", 2:"id1"}})

