import pandas as pd


def compare_blocks(a: pd.DataFrame, b: pd.DataFrame):
    assert all(
        a.klinker_block.to_pairs().sort_index()
        == b.klinker_block.to_pairs().sort_index()
    )
