import pandas as pd
from collections import OrderedDict
import pytest
from klinker import KlinkerBlockManager
from klinker.eval import Evaluation
from eche import PrefixedClusterHelper


@pytest.fixture()
def gold():
    return PrefixedClusterHelper(
        [
            {"A:1", "B:1"},
            {"A:2", "B:2"},
            {"A:3", "B:3"},
            {"A:4", "B:4"},
            {"A:5", "B:5"},
            {"A:6", "B:6"},
            {"A:7", "B:7"},
        ],
        ds_prefixes=OrderedDict(zip(["A", "B"], ["A:", "B:"])),
    )


@pytest.fixture()
def example():
    block = KlinkerBlockManager.from_pandas(
        pd.DataFrame(
            [
                [{"A:1", "A:2"}, {"B:1"}],
                [{"A:3", "A:2", "A:4"}, {"B:4", "B:5"}],
                [{"A:6"}, {"B:5", "B:7"}],
            ],
            columns=["A", "B"],
        )
    )
    return block


@pytest.fixture()
def example_nothing_found():
    block = KlinkerBlockManager.from_pandas(
        pd.DataFrame(
            [
                [{"A:1", "A:2"}, {"B:4"}],
                [{"A:3", "A:2", "A:4"}, {"A:1", "A:5"}],
                [{"A:6"}, {"A:5", "A:7"}],
            ],
            columns=["A", "B"],
        )
    )
    return block


@pytest.fixture()
def example_many_duplicates():
    block = KlinkerBlockManager.from_pandas(
        pd.DataFrame(
            [
                [{"A:1", "A:2"}, {"B:1", "B:4"}],
                [{"A:1", "A:2"}, {"B:1"}],
                [{"A:1", "A:3", "A:2", "A:4"}, {"B:1", "B:5"}],
                [{"A:1", "A:3", "A:2", "A:4"}, {"B:1", "B:5"}],
                [{"A:1", "A:3", "A:4"}, {"B:1", "B:5"}],
                [{"A:1", "A:3", "A:4"}, {"B:1", "B:4", "B:5"}],
                [{"A:1", "A:3", "A:4", "A:5"}, {"B:1", "B:4"}],
                [{"A:6"}, {"B:5", "B:7"}],
            ],
            columns=["A", "B"],
        )
    )
    return block


def test_quality(example, gold):
    e = Evaluation.from_blocks_and_gold(
        example, gold, left_data_len=8, right_data_len=7
    )
    assert e.true_positive == 2
    assert e.false_negative == 5
    assert e.false_positive == 8
    assert e.recall == pytest.approx(0.2857142857142857)
    assert e.precision == pytest.approx(0.2)
    assert e.f_measure == pytest.approx(0.23529411764705882)
    assert e.reduction_ratio == pytest.approx(0.8214285714285714)
    assert e.h3r == pytest.approx(0.42396313364055294)


def test_quality_nothing(example_nothing_found, gold):
    e = Evaluation.from_blocks_and_gold(
        example_nothing_found, gold, left_data_len=8, right_data_len=7
    )
    assert e.true_positive == 0
    assert e.false_negative == 7
    assert e.false_positive == 10
    assert e.recall == pytest.approx(0.0)
    assert e.precision == pytest.approx(0.0)
    assert e.f_measure == pytest.approx(0.0)
    assert e.reduction_ratio == pytest.approx(0.8214285714285714)
    assert e.h3r == pytest.approx(0.0)


def test_quality_duplicates(example_many_duplicates, gold):
    """Recall and true positive should be same as test_quality."""
    e = Evaluation.from_blocks_and_gold(
        example_many_duplicates, gold, left_data_len=8, right_data_len=7
    )
    assert e.true_positive == 2
    assert e.false_negative == 5
    assert e.recall == pytest.approx(0.2857142857142857)
