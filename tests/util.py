from klinker import KlinkerBlockManager


def assert_block_eq(
    one: KlinkerBlockManager,
    other: KlinkerBlockManager,
):
    assert len(one) == len(other)
    assert all(one.blocks.columns == other.blocks.columns)
    for blk_name in one.blocks.index:
        for col in one.blocks.columns:
            one_comp = set(one[blk_name][col].compute().values[0])
            other_comp = set(other[blk_name][col].compute().values[0])
            assert one_comp == other_comp
