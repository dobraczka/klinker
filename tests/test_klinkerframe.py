from typing import Dict, List, Tuple

import pandas as pd
import pytest
from strawman import dummy_df, dummy_triples

from klinker.data import KlinkerDaskFrame, KlinkerPandasFrame, KlinkerTriplePandasFrame


@pytest.fixture
def example() -> Dict:
    df = dummy_df((10, 3), columns=["colA", "colB", "colC"])
    return df.reset_index().to_dict()


@pytest.fixture
def concat_example() -> Tuple[KlinkerPandasFrame, List[List[str]]]:
    #df = dummy_df((5, 3), columns=["colA", "colB", "colC"])
    df = pd.DataFrame()
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
    print(df)
    return KlinkerPandasFrame.from_df(df, table_name="A", id_col="id"), expected


@pytest.fixture
def triple_example() -> Dict:
    return dummy_triples(10, relation_triples=False).to_dict()



@pytest.mark.parametrize("use_dask", [False, True])
def test_klinkerframe(example, use_dask):
    name = "A"
    id_col = "index"
    kf = KlinkerPandasFrame(table_name=name, id_col=id_col, data=example)
    if use_dask:
        kf = KlinkerDaskFrame.from_klinker_frame(kf, npartitions=2)
    assert kf.table_name == name
    assert kf.id_col == id_col


    # check that metadata is present after transformation
    transformed = kf[[kf.id_col, kf.columns[1], kf.columns[2]]]
    assert len(transformed.index) > 0
    assert len(transformed.columns) == 3
    assert transformed.table_name == name
    assert transformed.id_col == id_col

    transformed = kf.reset_index()
    assert len(transformed.index) > 0
    assert len(transformed.columns) == 5
    assert transformed.table_name == name
    assert transformed.id_col == id_col

    # check non-id-columns
    assert kf.non_id_columns == ["colA", "colB", "colC"]


@pytest.mark.parametrize("use_dask", [False, True])
def test_klinker_triple_frame(triple_example, use_dask):
    name = "A"
    id_col = "head"
    ktf = KlinkerTriplePandasFrame(data=triple_example, table_name=name, id_col=id_col)
    if use_dask:
        ktf = KlinkerDaskFrame.from_klinker_frame(ktf, npartitions=2)
    assert ktf.table_name == name
    assert ktf.id_col == id_col
    assert ktf.non_id_columns == ["tail"]

    new_column_name = "_merged_text"
    concated = ktf.concat_values(new_column_name=new_column_name)
    assert concated.table_name == name
    assert concated.id_col == id_col
    assert concated.non_id_columns == [new_column_name]
    if use_dask:
        concated = concated.compute()
        assert concated.table_name == name
        assert concated.id_col == id_col
        assert concated.non_id_columns == [new_column_name]
    assert len(concated[concated.id_col].unique()) == len(concated)

@pytest.mark.parametrize("use_dask", [False, True])
def test_concat(concat_example, use_dask):
    kf, expected = concat_example
    new_column_name = "_merged_text"
    if use_dask:
        kf = KlinkerDaskFrame.from_klinker_frame(kf, npartitions=2)
        concat_kf = kf.concat_values(new_column_name=new_column_name).compute()
    else:
        concat_kf = kf.concat_values(new_column_name=new_column_name)
    assert concat_kf.id_col == kf.id_col
    assert concat_kf.table_name == kf.table_name
    assert len(concat_kf.columns) == 2
    assert concat_kf[concat_kf.non_id_columns].values.tolist() == expected
