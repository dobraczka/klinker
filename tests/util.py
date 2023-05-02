import pandas as pd


def compare_blocks(a: pd.DataFrame, b: pd.DataFrame):
    assert (
        a.klinker_block.to_pairs().sort_index().values
        == b.klinker_block.to_pairs().sort_index().values
    ).all()
