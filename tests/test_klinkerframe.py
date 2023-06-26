from typing import Dict, List, Tuple

import pandas as pd
import pytest
from strawman import dummy_df, dummy_triples

from klinker.data import KlinkerFrame, KlinkerTripleFrame


@pytest.fixture
def example() -> Dict:
    df = dummy_df((10, 3), columns=["colA", "colB", "colC"])
    return df.reset_index().to_dict()


@pytest.fixture
def concat_example() -> Tuple[KlinkerFrame, List[str], List[List[str]]]:
    df = dummy_df((5, 3), columns=["colA", "colB", "colC"])

    wanted_cols = ["wc1", "wc2"]
    column_values = (
        ["Join", "Example", "more", "abc", "123"],
        ["me", "Text", "examples", "def", "456"],
    )
    for col_name, col_val in zip(wanted_cols, column_values):
        df[col_name] = col_val
    expected = [
        [" ".join([left, right])]
        for left, right in zip(column_values[0], column_values[1])
    ]
    df["id"] = [f"e{idx}" for idx in range(5)]
    return KlinkerFrame.from_df(df, name="A", id_col="id"), wanted_cols, expected


@pytest.fixture
def triple_example() -> Dict:
    return dummy_triples(10, relation_triples=False).to_dict()



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

def test_concat(concat_example):
    kf, wanted_cols, expected = concat_example
    concat_kf = kf.concat_values(wanted_cols)
    assert concat_kf.id_col == kf.id_col
    assert concat_kf.name == kf.name
    assert len(concat_kf.columns) == 2
    assert concat_kf[concat_kf.non_id_columns].values.tolist() == expected

