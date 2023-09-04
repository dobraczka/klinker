from typing import Dict, List, Tuple

import pandas as pd
import pytest
from strawman import dummy_df, dummy_triples
from test_blockers import example_tables, example_triples  # noqa: isort

from klinker.data import (
    KlinkerDaskFrame,
    KlinkerPandasFrame,
    KlinkerTriplePandasFrame,
    from_klinker_frame,
)


@pytest.fixture
def example() -> Dict:
    df = dummy_df((10, 3), columns=["colA", "colB", "colC"])
    return df.reset_index().to_dict()


@pytest.fixture
def concat_example() -> Tuple[KlinkerPandasFrame, List[Tuple[str, str]]]:
    df = pd.DataFrame()
    wanted_cols = ["wc1", "wc2"]
    column_values = (
        ["Join", "Example", "more", "abc", "123"],
        ["me", "Text", "examples", None, 456],
    )
    for col_name, col_val in zip(wanted_cols, column_values):
        df[col_name] = col_val
    df["id"] = [f"e{idx}" for idx in range(5)]
    expected = [
        (eid, " ".join([str(left), str(right)]).replace(" None", ""))
        for eid, left, right in zip(df["id"], column_values[0], column_values[1])
    ]
    return KlinkerPandasFrame.from_df(df, table_name="A", id_col="id"), expected


@pytest.fixture
def concat_triple_example(
    example_triples, example_tables
) -> Tuple[KlinkerPandasFrame, List[Tuple[str, str]]]:
    ta, _, _, _ = example_triples
    expected = [
        ("a1", "John McExample 11-12-1973 USA Engineer"),
        ("a2", "Maggie Smith 02-02-1983 USA Scientist"),
        ("a3", "Rebecca Smith 04-12-1990 Bulgaria Chemist"),
        ("a4", "Nushi Devi 14-03-1990 India"),
        ("a5", "Grzegorz Brzęczyszczykiewicz 02-04-1970 Poland Soldier"),
    ]
    return ta, expected


@pytest.fixture
def triple_example() -> Dict:
    return dummy_triples(10, relation_triples=False).to_dict()


@pytest.mark.parametrize("use_dask", [False, True])
def test_klinkerframe(example, use_dask):
    name = "A"
    id_col = "index"
    kf = KlinkerPandasFrame(table_name=name, id_col=id_col, data=example)
    if use_dask:
        kf = from_klinker_frame(kf, npartitions=2)
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
        ktf = from_klinker_frame(ktf, npartitions=2)
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


@pytest.mark.parametrize("example", ["concat_example", "concat_triple_example"])
@pytest.mark.parametrize("use_dask", [False, True])
def test_concat(example, use_dask, request):
    kf, expected = request.getfixturevalue(example)
    new_column_name = "_merged_text"
    if use_dask:
        kf = from_klinker_frame(kf, npartitions=2)
        concat_kf = kf.concat_values(new_column_name=new_column_name).compute()
    else:
        concat_kf = kf.concat_values(new_column_name=new_column_name)
    assert isinstance(concat_kf, pd.Series)
    assert concat_kf.name == kf.table_name
    edict = dict(expected)
    for eid, val in concat_kf.items():
        assert sorted(val.split(" ")) == sorted(edict[eid].split(" "))
