from typing import Dict, List, Tuple

import pytest

from klinker.data import KlinkerBlockManager


@pytest.fixture
def block_example() -> Tuple[
    KlinkerBlockManager,
    Dict[int, int],
    float,
    List[Tuple[int, int]],
    Tuple[Dict[int, str], Dict[int, str]],
]:
    fake_ids = "abcdefghijklmnopqrstuvwxyz"
    return (
        KlinkerBlockManager(
            blocks={
                2: ({3}, {3, 5}),
                4: ({4}, {5}),
                5: ({4}, {4}),
                6: ({4, 5}, {4}),
                7: ({5}, {4}),
                10: ({1, 2}, {2}),
                11: ({2}, {2}),
            },
            dataset_names=("A", "B"),
        ),
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
        (
            {int_id: f"l{fake_ids[int_id]}" for int_id in range(1, 6)},
            {int_id: f"r{fake_ids[int_id]}" for int_id in range(1, 6)},
        ),
    )


@pytest.fixture
def block_combine_example(
    block_example,
) -> Tuple[KlinkerBlockManager, KlinkerBlockManager, KlinkerBlockManager]:
    block = block_example[0]
    ds_names = ("A", "B")
    other_block = KlinkerBlockManager({6: block[6]}, ds_names)
    other_block[4] = ({1, 2}, {5})
    other_block[20] = ({5}, {7})
    expected = block.copy()
    expected[4] = ({1, 2, 4}, {5})
    expected[20] = ({5}, {7})
    return (
        block,
        other_block,
        expected,
    )


def test_safety_checks():
    # check tuple length
    with pytest.raises(ValueError):
        KlinkerBlockManager({2: ({1, 2}, {3}, {4})}, ("A", "B"))
    with pytest.raises(ValueError):
        KlinkerBlockManager(
            {2: ({1, 2}, {3})}, ("A", "B", "C"), ({1: "1", 2: "2"}, {3: "3"})
        )
    with pytest.raises(ValueError):
        KlinkerBlockManager(
            {2: ({1, 2}, {3})}, ("A", "B", "C"), ({1: "1", 2: "2"}, {3: "3"})
        )

    # check consistent id mappings
    with pytest.raises(ValueError):
        KlinkerBlockManager(
                {2: ({1, 2}, {3})}, ("A", "B"), ({1: "1"}, {3: "3"})
        )
    with pytest.raises(ValueError):
        KlinkerBlockManager(
                {2: ({1, 2}, {3})}, ("A", "B"), ({1: "1", 2: "2"}, {})
        )

def test_block_eq(block_example):
    other = KlinkerBlockManager(
        blocks={
            2: ({3}, {3, 5}),
            4: ({4}, {5}),
            5: ({4}, {4}),
            6: ({4, 5}, {4}),
            7: ({5}, {4}),
            10: ({1, 2}, {2}),
            11: ({2}, {2}),
        },
        dataset_names=("A", "B"),
    )
    example = block_example[0]
    assert example == other
    example.id_mappings = block_example[4]
    other.id_mappings = block_example[4]
    assert example == other


def test_block_copy(block_example):
    assert block_example[0].copy() == block_example[0]


def test_block_statistics(block_example):
    block, expected_sizes, mean_size, _, _ = block_example
    assert len(block) == len(block.blocks)
    assert expected_sizes == block.block_sizes.to_dict()
    assert mean_size == pytest.approx(block.mean_block_size)


def test_block_to_pairs(block_example):
    block, _, _, pairs, _ = block_example
    assert pairs == list(block.to_pairs())


def test_combine_blocks(block_combine_example):
    block, other_block, expected = block_combine_example
    assert KlinkerBlockManager.combine(block, other_block) == expected
    block.dataset_names = ("OTHER", "THANBEFORE")
    with pytest.raises(ValueError):
        KlinkerBlockManager.combine(block, other_block)
