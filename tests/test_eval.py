import pandas as pd
import pytest

from klinker.eval_metrics import Evaluation


@pytest.fixture
def example():
    block = pd.DataFrame(
        [[[1, 2], [1]], [[3, 2, 4], [4, 5]], [[6], [5, 7]]], columns=["A", "B"]
    )
    gold = pd.DataFrame(
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]], columns=["A", "B"]
    )
    return block, gold

@pytest.fixture
def example_nothing_found():
    block = pd.DataFrame(
        [[[1, 2], [4]], [[3, 2, 4], [1, 5]], [[6], [5, 7]]], columns=["A", "B"]
    )
    gold = pd.DataFrame(
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]], columns=["A", "B"]
    )
    return block, gold


def test_quality(example):
    block, gold = example
    e = Evaluation(block, gold, left_data_len=8, right_data_len=7)
    assert e.true_positive == 2
    assert e.false_negative == 5
    assert e.false_positive == 8
    assert e.recall == pytest.approx(0.2857142857142857)
    assert e.precision == pytest.approx(0.2)
    assert e.f_measure == pytest.approx(0.23529411764705882)
    assert e.reduction_ratio == pytest.approx(0.8214285714285714)
    assert e.h3r == pytest.approx(0.42396313364055294)

def test_quality_nothing(example_nothing_found):
    block, gold = example_nothing_found
    e = Evaluation(block, gold, left_data_len=8, right_data_len=7)
    assert e.true_positive == 0
    assert e.false_negative == 7
    assert e.false_positive == 10
    assert e.recall == pytest.approx(0.0)
    assert e.precision == pytest.approx(0.0)
    assert e.f_measure == pytest.approx(0.0)
    assert e.reduction_ratio == pytest.approx(0.8214285714285714)
    assert e.h3r == pytest.approx(0.0)
