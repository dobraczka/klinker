from typing import Dict, List, Tuple
from dataclasses import dataclass

import dask.dataframe as dd
import pandas as pd
import pytest

from klinker.data.blocks import KlinkerBlockManager

@dataclass
class Example:
    blocks:KlinkerBlockManager
    expected_sizes:Dict[int, int]
    mean_size:float
    pairs: List[Tuple[int, int]]
    blocks_dict: Dict[int,Tuple[List[int],List[int]]]
    dataset_names: Tuple[str,str]

@pytest.fixture
def block_example() -> Example:
    # fake_ids = "abcdefghijklmnopqrstuvwxyz"
    blocks_dict = {
        2: ([3], [3, 5]),
        4: ([4], [5]),
        5: ([4], [4]),
        6: ([4, 5], [4]),
        7: ([5], [4]),
        10: ([1, 2], [2]),
        11: ([2], [2]),
    }
    ds_names = ("A","B")
    return Example(
        KlinkerBlockManager.from_dict(blocks_dict, dataset_names=ds_names),
        {2: 3, 4: 2, 5: 2, 6: 3, 7: 2, 10: 3, 11: 2},
        2.4285714285714284,
        [
            (3, 3),
            (3, 5),
            (4, 5),
            (4, 4),
            (4, 4),
            (5, 4),
            (5, 4),
            (1, 2),
            (2, 2),
            (2, 2),
        ],
        blocks_dict,
        ds_names,
        # (
        #     {int_id: f"l{fake_ids[int_id]}" for int_id in range(1, 6)},
        #     {int_id: f"r{fake_ids[int_id]}" for int_id in range(1, 6)},
        # ),
    )


@pytest.fixture
def block_combine_example(
    block_example,
) -> Tuple[KlinkerBlockManager, KlinkerBlockManager, KlinkerBlockManager]:
    blocks_dict = block_example.blocks_dict
    other_block_dict = {6:blocks_dict[6],4:([1, 2], [5]),20:([5], [7])}
    expected = blocks_dict.copy()
    expected[4] = ([1, 2, 4], [5])
    expected[20] = ([5], [7])
    return (
        block_example.blocks,
        KlinkerBlockManager.from_dict(other_block_dict,block_example.dataset_names),
        KlinkerBlockManager.from_dict(expected,block_example.dataset_names),
    )


# def test_safety_checks():
#     # check tuple length
#     with pytest.raises(ValueError):
#         KlinkerBlockManager({2: ({1, 2}, {3}, {4})}, ("A", "B"))
#     with pytest.raises(ValueError):
#         KlinkerBlockManager(
#             {2: ({1, 2}, {3})}, ("A", "B", "C"), ({1: "1", 2: "2"}, {3: "3"})
#         )
#     with pytest.raises(ValueError):
#         KlinkerBlockManager(
#             {2: ({1, 2}, {3})}, ("A", "B", "C"), ({1: "1", 2: "2"}, {3: "3"})
#         )

#     # check consistent id mappings
#     with pytest.raises(ValueError):
#         KlinkerBlockManager({2: ({1, 2}, {3})}, ("A", "B"), ({1: "1"}, {3: "3"}))
#     with pytest.raises(ValueError):
#         KlinkerBlockManager({2: ({1, 2}, {3})}, ("A", "B"), ({1: "1", 2: "2"}, {}))


def test_block_eq(block_example):
    other = KlinkerBlockManager.from_dict(
        block_example.blocks_dict,
        dataset_names=block_example.dataset_names,
    )
    example = block_example.blocks
    assert example == other
    blocks_dict = block_example.blocks_dict
    blocks_dict[4] = ([4,6], [5])
    other = KlinkerBlockManager.from_dict(
        blocks_dict,
        dataset_names=block_example.dataset_names,
    )
    assert example != other


# def test_block_copy(block_example):
#     assert block_example[0].copy() == block_example[0]


def test_block_statistics(block_example):
    block = block_example.blocks
    assert len(block) == len(block.blocks)
    assert block_example.expected_sizes == block.block_sizes.to_dict()
    assert block_example.mean_size == pytest.approx(block.mean_block_size)


def test_block_to_pairs(block_example):
    block = block_example.blocks
    pairs = block_example.pairs
    assert sorted(pairs) == sorted(list(block.all_pairs()))
    assert sorted(list(set(pairs))) == sorted(list(block.all_pairs(remove_duplicates=True)))


def test_combine_blocks(block_combine_example):
    block, other_block, expected = block_combine_example
    assert KlinkerBlockManager.combine(block, other_block) == expected
    block.blocks.columns = ("OTHER", "THANBEFORE")
    with pytest.raises(ValueError):
        KlinkerBlockManager.combine(block, other_block)
