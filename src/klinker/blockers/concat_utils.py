import dask.dataframe as dd
import pandas as pd


def is_triple_df(df, assumed_columns=["head", "relation", "tail"]) -> bool:
    if len(df.columns) == len(assumed_columns):
        return all(df.columns == assumed_columns)
    return False


def concat_values(df, id_col: str = "head", remove_duplicates: bool = True):
    def _inner_concat_no_duplicates(grp):
        return " ".join(grp.astype(str).str.strip().unique()).strip()

    def _inner_concat_with_duplicates(grp):
        return " ".join(grp.astype(str).values).strip()

    agg_fun = (
        _inner_concat_no_duplicates
        if remove_duplicates
        else _inner_concat_with_duplicates
    )
    df = df.fillna("")
    if is_triple_df(df):
        if isinstance(df, dd.DataFrame):
            return (
                df[[id_col, df.columns[2]]]
                .groupby(id_col)[df.columns[2]]
                .apply(
                    agg_fun,
                    meta=pd.Series([], name=df.columns[2], dtype="str"),
                )
            )
        else:
            return (
                df[[id_col, df.columns[2]]].groupby(id_col).agg(agg_fun)[df.columns[2]]
            )
        if isinstance(df, dd.DataFrame):
            return df.set_index(id_col).apply(
                agg_fun, axis=1, meta=pd.Series([], dtype="str")
            )
    return df.set_index(id_col).apply(agg_fun, axis=1)
